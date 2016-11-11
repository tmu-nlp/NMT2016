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


