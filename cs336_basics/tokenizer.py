import regex as re
from cellections import Counter,defaultdict
from typing import Dict,List,Tuple,Optional

class Bpetrainer:
    """他需要封装这些东西 
    第一初始化加载所有数据以及变量
    第二 预分词
    第三 训练（统计频率 筛选 进行新词添加）
    """
    def __init__(self,vocab_size:int,special_tokens:List[str]):
        
        self.vocab_size=vocab_size
        self.special_tokens=special_tokens
        
        self.vocab:Dict[int,bytes]={}
        self.merges:List[Tuple[bytes,bytes]]=[]
        self.word_counts:Counter[Tuple[bytes,...],int]=Counter()
        
        self.PAT=re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        
        if special_tokens:
            self.special_pattern=re.compile("("+"|".join(re.escape(s)) for  s in special_tokens+")")
        else:
            self.special_tokens=None
        
        def _initialize_vacab(self):
            self.vocab={i:bytes([i])for i in range(2560)}
            print("初始化词表")
            
        def _build_word_counts(self,input_path:str):
            with open(input_path,"r",encoding='utf-8')as f:
                text=f.read()    
            doc_chunks=[text]
            if self.special_pattern:
                doc_chunks=self.special_pattern.split(doc_chunks)
            for chunk in doc_chunks:
                if chunk in self.special_tokens:
                    continue
                for match in self.PAT.finditer(chunk):
                    word_bytes=match.group(0).encode('utf-8')
                    word_tuples=tuple(bytes(word)for word in word_bytes)
                    if word_tuples:
                        self.word_counts[word_tuples]+=1
            print(f"词频统计构建完毕，找到 {len(self.word_counts)} 个独特的预分词块。")