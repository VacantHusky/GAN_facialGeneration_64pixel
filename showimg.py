import numpy as np
from PIL import Image
import tensorflow as tf
import tensorflow.contrib.slim as slim
import os,time

os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'
tf.set_random_seed(20190415)# 设置随机种子

img_W=64
img_H=img_W
img_L=3
batch_size=100
noise_dim = 100

if not os.path.exists('./testimgs/'):
    os.mkdir('./testimgs/')

def save_images(images,imgname,shape=(10,10)):
    images = np.uint8(images*255)
    # new_imgsize=img_W*10
    new_im = Image.new('RGB',(img_W*shape[1],img_H*shape[0]),(255,255,255))
    n=0
    for i in range(0,img_H*shape[0]-1,img_H):
        for j in range(0,img_W*shape[1]-1,img_W):
            try:
                new_im.paste(Image.fromarray(images[n]),(j,i))
            except:
                break
            n+=1
    new_im.save(imgname,'PNG')
    #new_im.show()

def deprocess_img(x):
    return (x+1.0) /2.0

def dc_generator(inputs, scope='dc_generator', reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        with slim.arg_scope([slim.fully_connected, slim.conv2d_transpose],activation_fn=None):
            net = slim.fully_connected(inputs, 4*4*512, scope='fc1')#将一维全链接层转换成二维网络
            net = tf.reshape(net, (-1, 4, 4, 512))#4*4*512
            net = tf.nn.relu(net, name='act1')#(?, 4,4,512)
            net = slim.batch_norm(net, scope='bn1')#(?, 4,4,512)

            net = slim.conv2d_transpose(net, 256, 5, stride=2, scope='convT3')
            net = tf.nn.relu(net, name='act3')
            net = slim.batch_norm(net, scope='bn3')

            net = slim.conv2d_transpose(net, 128, 5, stride=2, scope='convT4')
            net = tf.nn.relu(net, name='act4')
            net = slim.batch_norm(net, scope='bn4')
            
            net = slim.conv2d_transpose(net, 64, 5, stride=2, scope='convT41')
            net = tf.nn.relu(net, name='act41')
            net = slim.batch_norm(net, scope='bn41')
            
            net = slim.conv2d_transpose(net, 3, 5, stride=2, scope='convT5')
            net = tf.tanh(net, name='tanh')
            return net

sample_noise = tf.placeholder(tf.float32, shape=[None, noise_dim])
inputs_fake = dc_generator(sample_noise)

sess = tf.Session()
saver = tf.train.Saver()
if os.path.exists('my_net_3/checkpoint'):         #判断模型是否存在
    ckpt = tf.train.get_checkpoint_state('my_net_3/')
    saver.restore(sess,ckpt.model_checkpoint_path)
    print('读取记录成功')
else:
    print('读取记录失败')

# ========================================
np.random.seed(20190419)
z_samples = np.random.uniform(-1.0, 1.0, [batch_size, noise_dim]).astype(np.float32)
np.random.seed(int(time.time()))

step = 20
np_zz = np.zeros((step*10, noise_dim))
np_zz[0] = z_samples[79]
np_zz_shape=np_zz.shape[0]
p_ = 0
zhexian = 10

for i in range(zhexian):
    chu_point = np_zz[p_]
    next_point = z_samples[70+i]
    for j in range(np_zz_shape//zhexian):
        if p_+1>=np_zz_shape:
            break
        for n_2 in range(noise_dim):
            np_zz[p_+1][n_2]=chu_point[n_2]+(j+1)/(np_zz_shape//zhexian)*(next_point[n_2]-chu_point[n_2])
        p_+=1
# ========================================

one_fake_imgs=[]
for i in range(step):
    random_ = np.random.uniform(-1.0, 1.0, [batch_size, noise_dim]).astype(np.float32)
    for j in range(10):
        random_[:10]=np_zz[0+i*10:10+i*10]
    run_ = sess.run(inputs_fake,feed_dict={sample_noise:random_})
    for j in range(10):
        one_fake_imgs.append(run_[j])
sess.close()

resize_=128
imgs_numpy = deprocess_img(np.array(one_fake_imgs))
save_images(imgs_numpy,"./examples.png",shape=(10,step))
imgs_numpy = np.uint8(imgs_numpy*255)
im=Image.fromarray(imgs_numpy[0]).resize((resize_, resize_))
images=[]
for img in imgs_numpy:
    images.append(Image.fromarray(img).resize((resize_, resize_)))
    # Image.fromarray(img).resize((resize_, resize_)).save(f'./testimgs/test_{len(images)}.png')
im.save('./testimgs/test.gif', save_all=True, append_images=images,loop=0,duration=100,comment=b"aaabb")

print('end')