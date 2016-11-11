###NMT実装で使う主なfunctions
- reshape(Variable, (N,M)) # 第一位置引数にshapeを変更したいVariable、第二位置引数に次元
- concat((Variable1, Variable2, ,,,,)) # concatしたいVariableをリストorタプルに入れて渡す 
- stack((Variable1, Variable2, ,,,,)) # stackしたいVariableをリストorタプルに入れて渡す
- scale
- broadcast_to
- softmax_cross_entropy
- sum

####reshapeについて
Variableの次元数を変更  
reshape(Variable, (次元数))で使う

```python
x = Variable(np.array([1,2,3,4,5], dtype=np.float32))
In: x.data
Out: array([0, 1, 2, 3, 4], dtype=int32)

reshaped = functions.reshape(x, (5,1))
In: reshaped.data
Out: array([[ 1.],
            [ 2.],
            [ 3.],
            [ 4.],
            [ 5.]], dtype=float32)
            

