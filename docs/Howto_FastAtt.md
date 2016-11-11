###Attention計算部分の高速化メモ

```python
att = Att(dec_hidden, enc_hiddens)
# Input: デコーダの出力(j-1)とエンコーダの出力(0~i)
# Output: (batch_size, hidden_size)のVariable

# dec_hiddenはVariable  
In: dec_hidden.data.shape  
Out: (batch_size, hidden_size)  

# enc_hiddensはVariableのリスト  
In: enc_hiddens  
Out: [<variable at 0x1042a92b0>,  
      <variable at 0x1042a9780>,  
      <variable at 0x1042b4400>]  
# enc_hiddensの長さは文長  
# それぞれのVariableは、dec_hiddenと同じshape  
# → enc_hiddensは(sentence_len, batch_size, hidden_size) リストだからshapeとかないけども...  
```
####enc_hiddensの作り方

```python
class Encoder(Chain):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__(
            word2embed = L.EmbedID(vocab_size, embed_size, ignore_label=-1),
            embed2hidden_for = L.LSTM(embed_size, hidden_size),
            embed2hidden_back = L.LSTM(embed_size, hidden_size),
            hidden2enc = L.Linear(hidden_size*2, hidden_size), 
            )   

    def __call__(self, sent): 

        self._reset_state()
        embed_states = list()
        hidden_back_states = list()
        hidden_states = list()
        enc_states = list()
    
        for word in sent:
            embed = F.tanh(self.word2embed(word)) # 非線形にするの忘れずに
            embed_states.append(embed)
        for embed in embed_states[::-1]:
            hidden_back = self.embed2hidden_back(embed)
            hidden_back_states.insert(0, hidden_back)
        for e, h_back in zip(embed_states, hidden_back_states):
            hidden_for = self.embed2hidden_for(embed)
            hidden_states.append(F.concat((hidden_for, hidden_back)))
    
        enc_states = [self.hidden2enc(hidden) for hidden in hidden_states]
        # ↑ここで一度重み行列をかけておく（事前計算に該当)
        
        return enc_states
```
