from PIL import Image #이미지 파일을 읽어서 객체로 반환하게 하여 해당 이미지를 사용할 수 있도록 만들어주는 라이브러리이다.
import numpy as np #이미지 파일을 배열 형태로 만들고 covolution과 같은 행렬 연산을 위해 사용한 라이브러리이다.
import matplotlib.pyplot as plt #배열 형태로 만들어진 이미지를 2차원 상에 출력하기 위해 사용하는 라이브러리이다.

origin_img = Image.open('Lenna.png') #show함수에서 원본 이미지를 출력하기 위해 흑백으로 conver하기 전인 컬러 이미지를 받는다.
#이때 Image.open함수를 이용하여 이미지 파일을 읽어올 수 있도록 한다.

img = Image.open('Lenna.png').convert('L')#이미지를 불러와서 흑백으로 convert해준다. 이때 L이 흑백으로 만들어주는 옵션이다.
img = np.array(img) #이미지를 convolution 처리하기 용이하도록 배열 형태로 변환한다.

LoG_3 = np.array([[0, -1, 0],
                    [-1, 4, -1],
                    [0, -1, 0]])
# 3*3 크기의 mask를 만들고 값은 수업의 LoG pdf 파일에 적힌 kernel [-1,2,1]을 이용한다.
# 이때 mask는 laplacian filter 형태로 만들어서 배열 형태의 img에 convolution 하도록 한다.

LoG_5 = np.array([[0, 0, -1, 0, 0],
                  [0, -1, -2, -1, 0],
                  [-1, -2, 16, -2, -1],
                  [0, -1, -2, -1, 0],
                  [0, 0, -1, 0, 0]])
# 5*5 크기의 mask를 만들고 값은 수업의 LoG pdf 파일에 적힌 kernel [-1,2,1]을 확장해서 만든다.
# 이때 mask는 laplacian filter 형태로 만들어서 배열 형태의 img에 convolution 하도록 한다.

#위 2개의 laplacian filter는 img에 대해 x축으로 한번, y축으로 한번 covolution한 값을 더하는 형태의 연산을 하게 된다.


def show(result, threshold_result, filter_size_name):
    #show함수는 threshold를 적용하지 않고 단순히 LoG만 적용한 result, threshold를 적용하여 LoG를 진행한 threshold_result
    #laplacian filter size 값을 받는 filter_size_name를 파라미터로 받는다.
    #show함수는 각 결과를 보여주는 함수이다. 원본 이미지, LoG만 적용하여 edge를 찾은 이미지, LoG에서 threshold를 적용하여 edge를 찾은 이미지 순서대로 결과를 보여주고 있다.

    if(filter_size_name == 3): #3*3 laplacian filter를 적용하여 나온 결과를 보여준다.
        print("원본 이미지")
        plt.imshow(origin_img)
        plt.show()

        print("threshold를 적용하지 않은 LoG_3 edge-detection 이미지")
        plt.imshow(result, cmap='gray')
        plt.show()

        print("threshold를 적용한 LoG_3 edge-detection 이미지")
        plt.imshow(threshold_result, cmap='gray')
        plt.show()

    else: #5*5 laplacian filter를 적용하여 나온 결과를 보여준다.
        print("원본 이미지")
        plt.imshow(origin_img)
        plt.show()

        print("threshold를 적용하지 않은 LoG_5 edge-detection 이미지")
        plt.imshow(result, cmap='gray')
        plt.show()

        print("threshold를 적용한 LoG_5 edge-detection 이미지")
        plt.imshow(threshold_result, cmap='gray')
        plt.show()

    print() #LoG_3 출력과 LoG_5 출력 간 간격을 두기 위한 코드이다.


def edge_detection(img, mask1, threshold):
    #edge_detection함수는 numpy에 의해 배열로 만들어진 이미지 파일을 받는 img와 위에서 지정한 laplacian filter 배열을 받는 mask1, threshold값을 받는 threshold를 파라미터로 받는다.

    img_shape= img.shape #배열 형태의 img의 height와 width를 (height, width)라는 튜플 형태로 img.shape가 반환하여 img_shape에 저장한다.

    filter_size = mask1.shape #LoG에서 사용할 laplacian filter의 height, width 정보를 튜플 형태로 받는다.
    filter_size_name = filter_size[0] #mask1.shape은 튜플 형태이고 height의 값과 width의 값이 동일하므로 0번째 index의 값이
    #2개의 laplacian filter 중에서 3*3을 사용하였는지 5*5를 사용하였는지 구분할 수 있는 값에 해당하므로 이 값을 filter_size_name에 저장한다.
    #이 filter_size_name 변수는 어떤 filter를 LoG를 할 때 적용하였는지를 알려주는 변수이다.
    #즉, 3*3 laplacian filter를 적용하였는지, 5*5 laplacian filter를 적용하였는지를 받는다.

    result_shape = tuple(np.array(img_shape) - np.array(filter_size) + 1)
    #result_shape은 튜플 img_shape을 배열 형태로 만든 것과 튜플 filter_size를 배열 형태로 만든 것을 각 index 값에 대해 차연산과 +1연산을 하여
    #LoG를 적용하여 나오는 edge-detect 결과 이미지를 저장할 이미지에 대한 shape 정보를 받는 변수이다.

    result1 = np.zeros(result_shape)
    #result1 배열은 LoG_* 배열로 img를 convolution한 결과를 담는 배열이다.
    #result1의 shape정보는 최종 convolution 결과를 담는 result 배열의 shape 정보와 동일하다.


    for h in range(0, result_shape[0]): #img의 height에 대해 LoG convolution을 적용한다.
        for w in range(0, result_shape[1]): #먼저 img의 width에 대해 LoG convolution을 적용한다.
            temp = img[h:h + filter_size[0], w:w + filter_size[1]]
            #img 배열 값이 convolution 과정 중에서 변동되면 안되므로 img 배열의 값을 담는 temp 배열을 선언하여 img 배열의 값을 담는다.
            #그리고 이 temp의 역할은 전체 이미지 내에서 convolution을 적용할 block 범위를 지정하고 img 배열에서 그 범위 안에 포함되는 img 배열의 값을 담고 있는 배열이다.
            #그래서 h+filter_size[0], w+filter_size[1]을 하여 전체 img 배열 내에서 각각 filter_size[0], filter_size[1]의 범위만큼 지정하고
            #img 배열의 해당 범위의 값을 temp 배열에 저장한다. 그래서 반복문이 돌 때 마다 계속해서 한 pixel만큼 block이 옮겨가면서 convolution이 진행된다.
            #이때 지정된 block 범위는 covolution의 행렬 연산을 할 수 있는 범위인 mask1의 shape 정보(가로, 세로 범위)와 동일하다. 그래야지만 행렬 연산이 가능하기 때문이다.

            result1[h][w] = np.abs(np.sum(temp * mask1))
            #LoG_*를 img에 대해 convolution 즉, 행렬 연산을 한다.
            #convolution 연산은 2차원 배열 temp와 mask1의 각각의 요소마다 곱연산을 수행하고
            #그 다음에 곱연산이 끝난 각각의 요소들의 합을 구한 다음 절댓값을 취해서
            #그 값을 result1[h][w]에 저장한다.


    result = result1
    #LoG_*에 대해 covolution한 값을 가지고 있는 result1의 값을 최종 convolution 값을 만들고 이를 result 변수에 저장한다.

    threshold_result = np.zeros(result_shape)
    #threshold를 적용한 LoG_*의 결과를 받는 배열이다. 그러므로 최종 결과를 담고 있는 result와 같은 shape 정보를 가져야한다.
    #그리고 새로운 결과 값을 저장해야하므로 threshold_result 배열의 전체 요소 값을 0으로 초기화 해준다.

    for i in range(0, len(threshold_result)):
        #위의 convolution과 마찬가지로 width부터 threshold 값과 result 결과와 비교하고 그 다음에 height에 대해서 threshold값과 비교한다.
        #다만 이때 threshold값과 비교할 대상은 원본 이미지 파일인 img 배열이 아닌 convolution한 결과를 담고 있는 result 배열과 비교한다.

        for j in range(0, len(threshold_result[0])):
            if(result[i][j] > threshold): #convolution한 결과 값이 threshold 값보다 크면 result[i][j]로 threshold_result 값을 바꾼다.
                threshold_result[i][j] = result[i][j]
            else: #convolution한 결과 값이 threshold 값보다 작으면 noise로 간주하고 넘어간다.
                continue

    show(result, threshold_result, filter_size_name) #결과 이미지를 확인하기 위해 show함수를 호출한다.

    return result, threshold_result


edge_detection(img, LoG_3, threshold=100)
#원본 이미지를 흑백으로 convert한 이미지에 대해 3*3 laplacian filter와 threshold가 100일 때의 LoG를 적용하여 edge detect를 한다.

edge_detection(img, LoG_5, threshold=300)
#원본 이미지를 흑백으로 convert한 이미지에 대해 5*5 laplacian filter와 threshold가 300일 때의 LoG를 적용하여 edge detect를 한다.
