###NMT実装で使う主なfunctions
- reshape(Variable, (N,M)) # 1つめの引数にshapeを変更したいVariable、
- concat((Variable1, Variable2, ,,,,)) # concatしたいVariableをタプルに入れて渡す 
- stack
- scale
- broadcast_to
- softmax_cross_entropy
- sum

####reshapeについて
Variableの次元数を変更（numpyのreshapeとちがって、返り値があるよ）
reshape(Variable, (次元数))で使う

```python
x = Variable(np.array([1,2,3,4,5], dtype=np.float32))
In: x.data
Out: array([0, 1, 2, 3, 4], dtype=int32)

reshaped = functions.reshape(
