import numpy as np
from PIL import Image
import tensorflow as tf
import tensorflow.contrib.slim as slim
import os,time

os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'
tf.set_random_seed(20190415)# 设置随机种子

img_W=64           # 图片尺寸
img_H=img_W
img_L=3
img_examples=202600# 训练集图片数量
batch_size=100     # 每次训练的图片数量
noise_dim = 100    # 生成网络的输入维度
chongfu=300        # 重复次数
show_every = 2000  # 每训练多少次展示生成的图片
save_every = 2000  # 每训练多少次保存模型
image_path='/home/wanghu/virtua/DATA/img_align_celeba/' # 训练集所在路径

np.random.seed(20190419)
z_samples = np.random.uniform(-1.0, 1.0, [batch_size, noise_dim]).astype(np.float32)
np.random.seed(int(time.time()))

# 返回一个目录下的所有文件名
def get_file_name(filepath):
    files =os.listdir(filepath) #采用listdir来读取所有文件
    #files.sort() #排序
    s= []                   #创建一个空列表
    for file_ in files:     #循环读取每个文件名
        if not os.path.isdir(filepath +file_):  #判断该文件是否是一个文件夹
            s.append(str(filepath +file_))  # 把当前文件名返加到列表里
    return s

img_examples = len(get_file_name(image_path))

# 函数的功能是将filename对应的图片文件读进来，并缩放到统一的大小
def _parse_function(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.cond(
        tf.image.is_jpeg(image_string),
        lambda: tf.image.decode_jpeg(image_string),
        lambda: tf.image.decode_png(image_string)
    )
    image_decoded = tf.image.convert_image_dtype(image_decoded, tf.float32)
    image_resized = tf.image.resize_images(image_decoded, [img_H, img_W])
    return image_resized, label

# 图片文件的列表
filenames = tf.constant(get_file_name(image_path))
# label[i]就是图片filenames[i]的label
labels = tf.constant(get_file_name(image_path))

# 此时dataset中的一个元素是(filename, label)
dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
# 此时dataset中的一个元素是(image_resized, label)
dataset = dataset.map(_parse_function)
 
# 此时dataset中的一个元素是(image_resized_batch, label_batch)
dataset = dataset.shuffle(buffer_size=1000).batch(batch_size).repeat(chongfu)
iterator = dataset.make_one_shot_iterator()
one_element = iterator.get_next()

def save_images(images,imgname,shape=(10,10)):
    images = np.uint8(images*255)
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

# 输入层
input_ph = tf.placeholder(tf.float32, shape=[None, img_W, img_H, img_L])
inputs = tf.divide(input_ph-0.5, 0.5)

# 判别网络
def dc_discriminator(inputs, scope='dc_discriminator', reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        with slim.arg_scope([slim.conv2d, slim.fully_connected], activation_fn=None):
            net = slim.conv2d(inputs, 64, 5, stride=2, scope='conv1')
            net = tf.nn.leaky_relu(net, alpha=0.2, name='act1')
            # net = slim.max_pool2d(net, 2, stride=2, scope='maxpool1')
            
            net = slim.conv2d(net, 128, 5, stride=2, scope='conv2')
            net = tf.nn.leaky_relu(net, alpha=0.2, name='act2')
            # net = slim.max_pool2d(net, 2, stride=2, scope='maxpool2')
            
            net = slim.conv2d(net, 256, 5, stride=2, scope='conv3')
            net = tf.nn.leaky_relu(net, alpha=0.2, name='act3')
            # net = slim.max_pool2d(net, 2, stride=2, scope='maxpool3')

            net = slim.conv2d(net, 512, 5, stride=2, scope='conv4')
            net = tf.nn.leaky_relu(net, alpha=0.2, name='act4')
            # net = slim.max_pool2d(net, 2, stride=2, scope='maxpool3')

            net = slim.flatten(net, scope='flatten')
            net = slim.fully_connected(net, 1, scope='fc5')
            # net = tf.nn.leaky_relu(net, alpha=0.01, name='act5')
            # net = slim.fully_connected(net, 1, scope='fc6')
            return net

# 生成网络
def dc_generator(inputs, scope='dc_generator', reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        with slim.arg_scope([slim.fully_connected, slim.conv2d_transpose],activation_fn=None):
            net = slim.fully_connected(inputs, 4*4*512, scope='fc1')#将一维全链接层转换成二维网络
            net = tf.reshape(net, (-1, 4, 4, 512))
            net = tf.nn.relu(net, name='act1')
            net = slim.batch_norm(net, scope='bn1')

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

# true_labels = tf.ones((batch_size, 1), dtype=tf.float32, name='true_labels')
true_labels = 1-tf.abs(tf.truncated_normal(
    (batch_size, 1),
    mean=0.0,
    stddev=0.26,
    dtype=tf.float32,
    name='true_labels'
))

# fake_labels = tf.zeros((batch_size, 1), dtype=tf.float32, name='fake_labels')
fake_labels = tf.abs(tf.truncated_normal(
    (batch_size, 1),
    mean=0.0,
    stddev=0.31,
    dtype=tf.float32,
    name='fake_labels'
))

def s_c_e_w_l(x, y):
    return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)

sample_noise = tf.placeholder(tf.float32, shape=[None, noise_dim])

inputs_fake = dc_generator(sample_noise)
logits_real = dc_discriminator(inputs) # 获得真实数据的分数
logits_fake = dc_discriminator(inputs_fake, reuse=True) # 获得生成数据的分数

loss_d_real = tf.reduce_mean(s_c_e_w_l(logits_real, true_labels))
loss_d_fake = tf.reduce_mean(s_c_e_w_l(logits_fake, fake_labels))
loss_g = tf.reduce_mean(s_c_e_w_l(logits_fake, true_labels))

d_total_error = loss_d_real + loss_d_fake # 判别网络`loss`
g_total_error = loss_g # 生成网络`loss`


# 我们使用adam来训练, 学习率是3e-4, beta1是0.5, beta2是0.999
opt = tf.train.AdamOptimizer(2e-4, beta1=0.5, beta2=0.999)

# 获得判别网络的可训练参数
dc_discriminator_params = tf.trainable_variables('dc_discriminator')
# 构造判别训练`op`
train_discriminator = opt.minimize(d_total_error, var_list=dc_discriminator_params)

# 获得生成网络可训练参数
dc_generator_params = tf.trainable_variables('dc_generator')
# 构造生成训练'op'
with tf.control_dependencies([train_discriminator]):
    train_generator = opt.minimize(g_total_error, var_list=dc_generator_params)


iter_count = tf.Variable(0,name='iter_count')#迭代次数
num_one = tf.constant(1)
new_iter_count = tf.add(iter_count, num_one)
iter_updata = tf.assign(iter_count,new_iter_count) #赋值，iter_count=new_iter_count

sess = tf.Session()

# 尝试读取之前的记录
#判断模型保存路径是否存在，不存在就创建
if not os.path.exists('my_net/'):
    os.mkdir('my_net/')
if not os.path.exists('imgs/'):
    os.mkdir('imgs/')
saver = tf.train.Saver()
if os.path.exists('my_net/checkpoint'):         #判断模型是否存在
    ckpt = tf.train.get_checkpoint_state('my_net/')
    saver.restore(sess,ckpt.model_checkpoint_path)
    print('读取记录成功')
else:
    sess.run(tf.global_variables_initializer())


for e in range(chongfu):
    num_examples = 0
    while num_examples < img_examples:
        num_examples += batch_size
        train_imgs, _ = sess.run(one_element)
        sess.run(iter_updata)#更新
        int_iter_count = sess.run(iter_count)
        x_r = train_imgs
        n = np.random.uniform(-1.0, 1.0, [batch_size, noise_dim]).astype(np.float32)
        loss_d,loss_g,fake_imgs,_=sess.run(
            [d_total_error, g_total_error,inputs_fake, train_generator],
            feed_dict={input_ph: x_r,sample_noise:n}
        )
        if int_iter_count%show_every == 1 or (num_examples>=img_examples and e+1==chongfu):
            print('Iter: {}, 鉴别器: {:.4f}, 生成器: {:.4f}'.format(int_iter_count, loss_d, loss_g))
            one_fake_imgs = sess.run(inputs_fake,feed_dict={sample_noise:z_samples})
            imgs_numpy = deprocess_img(one_fake_imgs)
            save_images(imgs_numpy,"./imgs/examples-"+str(int_iter_count)+".png")
            print()
        if int_iter_count%save_every == 1 or (num_examples>=img_examples and e+1==chongfu):
            #保存
            save_path = saver.save(sess, "my_net/save_net.ckpt",global_step=int_iter_count)
            print("保存到路径: ", save_path)
sess.close()
print('end')