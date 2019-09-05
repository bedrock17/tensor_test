import tensorflow as tf

# x_data = np.array([[0, 0], [1, 0], [1, 1], [0, 0], [0, 0], [0, 1]])

import myImageLoader as IL
import numpy as np


x_lst = []
y_lst = []

def appendTestData(fileName, ans):
    global x_lst
    global y_lst
    x_lst.append(IL.loadImageArray("mynnimg/" + fileName))
    y_lst.append(ans)

def makeAnsArr(idx: int):
    ans = [0] * 10
    ans[idx] = 1
    return ans

appendTestData("1_1.png", makeAnsArr(1))
appendTestData("1_2.png", makeAnsArr(1))
appendTestData("1_3.png", makeAnsArr(1))
appendTestData("2.png", makeAnsArr(2))
appendTestData("3.png", makeAnsArr(3))
appendTestData("4.png", makeAnsArr(4))
appendTestData("4_2.png", makeAnsArr(4))
appendTestData("7.png", makeAnsArr(7))
appendTestData("8.png", makeAnsArr(8))
appendTestData("8_2.png", makeAnsArr(8))
appendTestData("8_3.png", makeAnsArr(8))
appendTestData("8_4.png", makeAnsArr(8))

x_data = np.array(x_lst)
y_data = np.array(y_lst)

#########
# 신경망 모델 구성
######
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

tmp = 256

W1 = tf.Variable(tf.random_normal([784, tmp], stddev=0.01))
L1 = tf.nn.relu(tf.matmul(X, W1))

W2 = tf.Variable(tf.random_normal([tmp, tmp], stddev=0.01))
L2 = tf.nn.relu(tf.matmul(L1, W2))

# W2_2 = tf.Variable(tf.random_normal([tmp, tmp], stddev=0.01))
# L2_2 = tf.nn.relu(tf.matmul(L2, W2_2))

# W2_3 = tf.Variable(tf.random_normal([tmp, tmp], stddev=0.01))
# L2_3 = tf.nn.relu(tf.matmul(L2_2, W2_3))


W3 = tf.Variable(tf.random_normal([tmp, 10], stddev=0.01))

# 최종적인 아웃풋을 계산합니다.
# 히든레이어에 두번째 가중치 W2와 편향 b2를 적용하여 3개의 출력값을 만들어냅니다.
model = tf.matmul(L2, W3)

# 텐서플로우에서 기본적으로 제공되는 크로스 엔트로피 함수를 이용해
# 복잡한 수식을 사용하지 않고도 최적화를 위한 비용 함수를 다음처럼 간단하게 적용할 수 있습니다.
cost = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=model))

optimizer = tf.train.AdamOptimizer(learning_rate=0.005)
train_op = optimizer.minimize(cost)


#########
# 신경망 모델 학습
######
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for step in range(2000):
    sess.run(train_op, feed_dict={X: x_data, Y: y_data})

    if (step + 1) % 10 == 0:
        print(step + 1, sess.run(cost, feed_dict={X: x_data, Y: y_data}))


#########
# 결과 확인
# 0: R 1: G, 2: B
######
prediction = tf.argmax(model, 1)
target = tf.argmax(Y, 1)

t1 = []
t1.append(IL.loadImageArray("mynnimg/a_1.png"))
t1.append(IL.loadImageArray("mynnimg/a_4.png"))
t1.append(IL.loadImageArray("mynnimg/a_7.png"))
t1.append(IL.loadImageArray("mynnimg/a_8.png"))
t2 = []
t2.append(makeAnsArr(1))
t2.append(makeAnsArr(4))
t2.append(makeAnsArr(7))
t2.append(makeAnsArr(8))

myx = t1
myy = t2

print('예측값:', sess.run(prediction, feed_dict={X: myx}))
print('실제값:', sess.run(target, feed_dict={Y: myy}))

is_correct = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도: %.2f' % sess.run(accuracy * 100, feed_dict={X: myx, Y: myy}))
