# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
x1 = np.random.rand( 100, 1 )      # 0~1까지 난수를 100개 만든다
x1 = x1 * 4 - 2                    # 값의 범위를 -2~2로 변경

x2 = np.random.rand( 100, 1 )     # x2에 대해서도 마찬가지
x2 = x2 * 4 - 2

y = 3 * x1 - 2 * x2 + 1

plt.subplot( 1, 2, 1 )
plt.scatter( x1, y, marker='+' )
plt.xlabel( 'x1' )
plt.ylabel( 'y' )

plt.subplot( 1, 2, 2 )
plt.scatter( x2, y, marker='+' )
plt.xlabel( 'x2' )
plt.ylabel( 'y' )

plt.tight_layout()
plt.show()
