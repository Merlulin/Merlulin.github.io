# 计算卷积后的维度

```python
def cal_h_w(hin, win, k_size, padding, stride):
    hout = (hin + 2 * padding - k_size) / stride + 1
    wout = (win + 2 * padding - k_size) / stride + 1
    return hout, wout
```