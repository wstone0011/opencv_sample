# 说明
```
1.
img = cv.imread('digits.png',0)
cells = [np.hsplit(row,100) for row in np.vsplit(img,50)]

digits.png 高1000,宽2000
被分割成50*100个图，每个图20*20大小

cells与img的元素对应关系如下：
cells[a][b][i][j] = img[20*a+i][20*b+j]
img[h][w] = cells[h/20][w/20][h%20][w%20]

例如：
cells[1][2][3][9] = img[20+3][20*2+9]=img[23][49] = 11

2.
train_cells = [ i[:50] for i in cells ]
test_cells = [ i[50:] for i in cells]

每个数字的前50个数据作为训练数据，后50个数据作为测试数据

```
