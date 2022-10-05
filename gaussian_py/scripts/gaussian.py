#!/usr/bin/env python3
import rospy
from itertools import product
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from gaussian_py.msg import kl_suzu
from gaussian_py.msg import kl_suzuki
from statistics import stdev
import time
import math
from codecs import utf_16_be_encode
from sklearn.feature_selection import mutual_info_classif

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import csv
import pprint
import collections
import itertools
import random

def flatten(l):
    for el in l:
        if isinstance(el, collections.abc.Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el
def M(mumu: np.array, sigsig: np.array, p, q):
                u = mumu / sigsig
                U1 = 1 / (1 + np.exp(p * (u - q)))
                return U1
def min_max(x,axis=None):
    min = x.min(axis=axis, keepdims=True)
    max = x.max(axis=axis, keepdims=True)
    result = (x-min)/(max-min)
    return result
semi = kl_suzu()
#tf = int
tf = 0
def klCallback1(msg):
    arg1 = 10
    arg2 = 27
    n = arg1
    #tf = int = 0
    global tf
    size=len(msg.KL)
    rospy.loginfo("size: %d", size)
    #X = np.empty([1,2])
    #y = np.empty([1,1])
    for i in range(0,size):
        if i==0:
            X=np.array([[msg.KL[i].kl_x,msg.KL[i].kl_y,msg.KL[i].kl_z]])
            y=np.array([msg.KL[0].kl_score])
        else:
            b=np.array([[msg.KL[i].kl_x,msg.KL[i].kl_y,msg.KL[i].kl_z]])
            X=np.vstack((X,b))
            c=np.array(msg.KL[i].kl_score)
            y=np.hstack((y,c))

    xx = []
    yy = []
    zzz = []
    for t in range(size):
        xx.append(X[t][0])
        yy.append(X[t][1])
        zzz.append(X[t][2])
    
    # Input space
    # datax = np.linspace(-1.5, 1.5, arg1) #p
    # datay = np.linspace(-0.75, 0.75, arg1) #q
    # dataz = np.linspace(0.8, 1.8, arg1)
    datax = np.linspace(-0.75, 0.75, arg1) #p
    datay = np.linspace(-0.5, 0.5, arg1) #q
    dataz = np.linspace(0.8, 1.8, arg1)
    data_x, data_y, data_z = np.meshgrid(datax, datay, dataz)
    xy_all = np.stack([data_x, data_y, data_z], 3)
    m = 0
    sample_index = np.array([[0, 0, 0], [0, 5, 0], [0, 9, 0], [5, 0, 0], [5, 5, 0], [5, 9, 0], [9, 0, 0], [9, 5, 0], [9, 9, 0], 
                [0, 0, 5], [0, 5, 5], [0, 9, 5], [5, 0, 5], [5, 5, 5], [5, 9, 5], [9, 0, 5], [9, 5, 5], [9, 9, 5],
                [0, 0, 9], [0, 5, 9], [0, 9, 9], [5, 0, 9], [5, 5, 9], [5, 9, 9], [9, 0, 9], [9, 5, 9], [9, 9, 9]])
    sample_quantity = len(sample_index)
    sample_index_x = sample_index[:, 0]
    sample_index_y = sample_index[:, 1]
    sample_index_z = sample_index[:, 2]
    ######################
    # ガウスカーネル(RBFカーネル)を関数化
    def kernel(x: np.array, x_prime: np.array, p, q, r):
        # if x == x_prime:
        #     delta = 1
        # else:
        #     delta = 0
        return p*np.exp(-1 * np.linalg.norm(x - x_prime)**2 / q) + r
    # データの定義
    xtrain = np.copy(datax[sample_index_x])
    ytrain = np.copy(datay[sample_index_y])
    ztrain = np.copy(dataz[sample_index_z])
    ztrain1 = np.zeros(arg2 * 3, dtype = 'int').reshape((arg2, 3))
    ztrain11 = np.zeros(arg2, dtype = 'int')
    xtest = np.copy(datax)
    ytest = np.copy(datay)
    ztest = np.copy(dataz)
    test_length = len(xtest)
    # 平均
    mu = []
    # 分散
    var = []
    ss = []
    kk = []
    L = []
    mu_ani = []
    var_ani = []
    # 各パラメータ値
    Theta_1 = 1
    Theta_2 = 0.5
    Theta_3 = 0.0
    # 以下，ガウス過程回帰の計算の基本アルゴリズム
    train_length = len(xtrain) #len():因数に指定したオブジェクトの長さや要素の数を取得する．
    # print('train_length', train_length)
    # トレーニングデータ同士のカーネル行列の下地を準備
    K = np.zeros((train_length, train_length)) # np.zeros((2,4)):2x4の2次元配列を生成
    xy = np.c_[xtrain, ytrain, ztrain]
    if size == 1:
        for x in range(train_length):  #range(stop): 指定した開始数から終了数までの連続した数値を要素として持つrenge型のオブジェクトを生成する．
            for x_prime in range(train_length):
                K[x, x_prime] = kernel(xy[x], xy[x_prime], Theta_1, Theta_2, Theta_3)
        # print('K', K)
        K_inv = np.linalg.inv(K)
        KK = list(flatten(K))
        with open('sisaku_K.csv', 'w', newline='') as ff:
            writer5 = csv.writer(ff)
            writer5.writerow(KK)
        # 内積はドットで計算
        zz = np.dot(np.linalg.inv(K), ztrain11)
        # print('xy_all', xy_all)
        # print('test_length', test_length)
        time_get = time.time()
        for x_test1 in range(test_length):
            for x_test2 in range(test_length):
                for x_test3 in range(test_length):
                    # テストデータとトレーニングデータ間のカーネル行列の下地を準備
                    k = np.zeros((train_length,)) 
                    s = kernel(xy_all[x_test1][x_test2][x_test3], xy_all[x_test1][x_test2][x_test3], Theta_1, Theta_2, Theta_3)
                    ss.append(s)
                    for x in range(train_length):
                        k[x] = kernel(xy[x], xy_all[x_test1][x_test2][x_test3], Theta_1, Theta_2, Theta_3)
                    kk.append(k)
                    # 内積はドットで計算して，平均値の配列に追加
                    # print('len(C_pC_p)', len(C_pC_p))
                    # print('x_test1', x_test1)
                    # print('x_test2', x_test2)
                    # print('x_test3', x_test3)
        time_out = time.time() - time_get
        # print('time_out', time_out)
        time_get = time.time()
        kk = np.array(kk)
        mu = np.matmul(kk, zz)
        mum = mu.tolist()
        with open('sisaku_mu.csv', 'w', newline='') as ff:
            writer3 = csv.writer(ff)
            writer3.writerow(mum)
        # JJ = np.matmul(kk, np.linalg.inv(K))
        time_out1 = time.time() - time_get
        # print('time_out1', time_out1)
        time_get = time.time()
        JJ = []
        for fgh in range(1000):
            # print('fgh', fgh)
            J = np.matmul(kk[fgh], K_inv)
            JJ.append(J)
        JJ = np.array(JJ)
        JJJ = list(flatten(JJ))
        with open('sisaku_JJ.csv', 'w', newline='') as ff:
            writer2 = csv.writer(ff)
            writer2.writerow(JJJ)
        time_out2 = time.time() - time_get
        # print('time_out2', time_out2)
        time_get = time.time()
        BB = []
        for hh in range(1000):
            BBBB = ss[hh] - np.matmul(JJ[hh, :], kk[hh, :])
            BB.append(BBBB)
        BB = np.array(BB)
        time_out3 = time.time() - time_get
        # print('time_out3', time_out3)
        time_get = time.time()
        C_pC_p = []
        for hh in range(1000):
            gg = BB[hh] + np.matmul(np.matmul(JJ[hh, :], K), JJ[hh, :])
            C_pC_p.append(gg)
        with open('sisaku_C_pC_p.csv', 'w', newline='') as ff:
            writer4 = csv.writer(ff)
            writer4.writerow(C_pC_p)
        C_pC_p = np.array([C_pC_p])
        timeout4 = time.time() - time_get
        # print('timeout4', timeout4)
        mu = mu.reshape(arg1, arg1, arg1)
        mu_min_iti = np.unravel_index(mu.argmin(), mu.shape)
        mu_min = np.amin(mu)
        # mu_minx = minx.index(min(minx))
        # mu_miny = miny.index(min(miny))
        # mu_minz = minz.index(min(minz))
        # print('JJ', JJ)
        # print('C_pC_p', C_pC_p)
        # JJ = np.array([JJ])
        # print('JJ', JJ)
        JJ = JJ.reshape(arg1, arg1, arg1, arg2)
        # print('JJ', JJ)
        # C_pC_p = np.array([C_pC_p])
        C_pC_p = C_pC_p.reshape(arg1, arg1, arg1)
        C_p_min = np.unravel_index(C_pC_p.argmin(), C_pC_p.shape)
        # print('C_p_min', C_p_min)
        std = np.sqrt(C_pC_p)
        U = M(mumu = mu, sigsig = std, p = 0.1, q = 10)
        # Figureを追加
        ##################################
        ##################################
        ##事後分布を作成し，事前分布に回す#####
        # print('okok2')
        new = y[size - 1]
        new_x = X[size - 1][0]
        new_y = X[size - 1][1]
        new_z = X[size - 1][2]
        m = m + 1
        #このnew_xとnew_yがdataxの何番目に相当するかの数字が欲しい
        #new_xとプロットされた点の差が一番小さいやつにする．
        minx = []
        miny = []
        minz = []
        for t in range(n):
            d1 = abs(new_x - datax[t])
            minx.append(d1)
            d2 = abs(new_y - datay[t])
            miny.append(d2)
            d3 = abs(new_z - dataz[t])
            minz.append(d3)
        min_x = minx.index(min(minx))
        min_y = miny.index(min(miny))
        min_z = minz.index(min(minz))
        GG = JJ[min_x][min_y][min_z].T / C_pC_p[min_x][min_y][min_z]
        G = np.matmul(K, GG.T)
        G = np.array([G])
        G = G.T
        mu_f = G * (new - mu[min_x][min_y][min_z])
        mu_f = list(itertools.chain.from_iterable(mu_f))
        # C_f = K - np.dot(G, np.dot(JJ[4].T, K))  # これだとうまく計算できていない
        G2 = np.array([np.matmul(JJ[min_x][min_y][min_z].T, K)])
        # # print('G2', G2)
        C_f = K - np.matmul(G, G2)
        with open('sisaku_mu_f.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(mu_f)
        with open('sisaku_C_f.csv', 'w', newline='') as ff:
            writer1 = csv.writer(ff)
            writer1.writerows(C_f)
        semi.kl_score = mu_min
        semi.kl_x = mu_min_iti[0]
        semi.kl_y = mu_min_iti[1]
        semi.kl_z = mu_min_iti[2]
        rospy.loginfo('ok')
        # semi.saiteki_kl = saiteki_KL
        # semi.sinrai_kl = lb_kl_sigma
        # semi.jizen_heikin = mu_f
        # semi.jizen_bunsan = C_f
    ################################
    if size != 1 and size != 0:
        # 新たな入力x_sin, 出力y_sin
        time_get = time.time()
        new = y[size - 1]
        new_x = X[size - 1][0]
        new_y = X[size - 1][1]
        new_z = X[size - 1][2]
        m = m + 1
        #このnew_xとnew_yがdataxの何番目に相当するかの数字が欲しい
        #new_xとプロットされた点の差が一番小さいやつにする．
        minx = []
        miny = []
        minz = []
        for t in range(n):
            d1 = abs(new_x - datax[t])
            minx.append(d1)
            d2 = abs(new_y - datay[t])
            miny.append(d2)
            d3 = abs(new_z - dataz[t])
            minz.append(d3)
        min_x = minx.index(min(minx))
        min_y = miny.index(min(miny))
        min_z = minz.index(min(minz))
        filename = 'sisaku_mu_f.csv'
        with open(filename, encoding='utf8') as f:
            csvreader = csv.reader(f)
            content = [row for row in csvreader]  # 各年のデータを要素とするリスト
        content = list(flatten(content))
        content1 = [float(s) for s in content]
        #####################################################
        filename2 = 'sisaku_C_f.csv'
        with open(filename2, encoding = 'utf8') as f:
            csvreader2 = csv.reader(f)
            content2 = [row for row in csvreader2]
        content2 = list(flatten(content2))
        content3 = [float(s) for s in content2]
        content3 = np.array(content3).reshape(27, 27).tolist()
        ######################################################
        filename3 = 'sisaku_mu.csv'
        with open(filename3, encoding='utf8') as f:
            csvreader3 = csv.reader(f)
            content4 = [row for row in csvreader3]  # 各年のデータを要素とするリスト
        content4 = list(flatten(content4))
        content5 = [float(s) for s in content4]
        content5 = np.array(content5).reshape(arg1, arg1, arg1)
        ######################################################
        filename4 = 'sisaku_C_pC_p.csv'
        with open(filename4, encoding='utf8') as f:
            csvreader4 = csv.reader(f)
            content6 = [row for row in csvreader4]  # 各年のデータを要素とするリスト
        content6 = list(flatten(content6))
        content7 = [float(s) for s in content6]
        content7 = np.array(content7).reshape(arg1, arg1, arg1)
        #########################################################
        filename5 = 'sisaku_JJ.csv'
        with open(filename5, encoding='utf8') as f:
            csvreader5 = csv.reader(f)
            content8 = [row for row in csvreader5]  # 各年のデータを要素とするリスト
        content8 = list(flatten(content8))
        content9 = [float(s) for s in content8]
        content9 = np.array(content9).reshape(arg1, arg1, arg1, arg2)
        #########################################################
        filename6 = 'sisaku_K.csv'
        with open(filename6, encoding='utf8') as f:
            csvreader6 = csv.reader(f)
            content10 = [row for row in csvreader6]  # 各年のデータを要素とするリスト
        content10 = list(flatten(content10))
        content11 = [float(s) for s in content10]
        content11 = np.array(content11).reshape(arg2, arg2)
        content11_inv = np.linalg.inv(content11)
        ######################################################
        GG = content9[min_x][min_y][min_z].T / content7[min_x][min_y][min_z]
        G = np.matmul(content11, GG.T)
        G = np.array([G])
        G = G.T
        mu_f1 = np.array([content1]) + (G * (new - content5[min_x][min_y][min_z])).T
        mu_f1 = list(itertools.chain.from_iterable(mu_f1))
        G2 = np.array([np.matmul(content9[min_x][min_y][min_z].T, content3)])
        C_f1 = content3 - np.dot(G, G2)
        #################################################
        with open('sisaku_mu_f.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(mu_f1)
        with open('sisaku_C_f.csv', 'w') as ff:
            writer1 = csv.writer(ff)
            writer1.writerows(C_f1)
        #################################################
        ##############################
        # 平均
        # 分散
        var = []
        ss = []
        C_pC_p = []
        kk = []
        zz = np.dot(content11_inv, mu_f1)
        for x_test1 in range(test_length):
            for x_test2 in range(test_length):
                for x_test3 in range(test_length):
                    # 内積はドットで計算して，平均値の配列に追加
                    k = np.zeros((train_length,)) 
                    s = kernel(xy_all[x_test1][x_test2][x_test3], xy_all[x_test1][x_test2][x_test3], Theta_1, Theta_2, Theta_3)
                    ss.append(s)
                    for x in range(train_length):
                        k[x] = kernel(xy[x], xy_all[x_test1][x_test2][x_test3], Theta_1, Theta_2, Theta_3)
                    kk.append(k)
                    
        kk = np.array(kk)
        mu = np.matmul(kk, zz)
        mum = mu.tolist()
        with open('sisaku_mu.csv', 'w', newline='') as ff:
            writer3 = csv.writer(ff)
            writer3.writerow(mum)
        # JJ = np.matmul(kk, np.linalg.inv(content11))
        JJ = []
        for fgh in range(1000):
            J = np.matmul(kk[fgh], content11_inv)
            JJ.append(J)
        JJ = np.array(JJ)
        JJJ = list(flatten(JJ))
        with open('sisaku_JJ.csv', 'w', newline='') as ff:
            writer2 = csv.writer(ff)
            writer2.writerow(JJJ)
        BB = []
        for hh in range(1000):
            BBBB = ss[hh] - np.matmul(JJ[hh, :], kk[hh, :])
            BB.append(BBBB)
        BB = np.array(BB)
        C_pC_p = []
        for hh in range(1000):
            gg = BB[hh] + np.matmul(np.matmul(JJ[hh, :], C_f1), JJ[hh, :])
            C_pC_p.append(gg)
        with open('sisaku_C_pC_p.csv', 'w', newline='') as ff:
            writer4 = csv.writer(ff)
            writer4.writerow(C_pC_p)
        C_pC_p = np.array([C_pC_p])
        # l = 0
        # m = m + 1
        # for h in range(n):
        #     H = (mu1[h] - data_y[h]) ** 2
        #     # print('H', H)
        #     l = l + H
        # L.append(l)
        mu = mu.reshape(arg1, arg1, arg1)
        # print('type(mu1)', type(mu1))
        JJ = np.array([JJ])
        JJ = JJ.reshape(arg1, arg1, arg1, arg2)
        C_pC_p = np.array([C_pC_p])
        C_pC_p = C_pC_p.reshape(arg1, arg1, arg1)
        mu_ani.append(mu)
        var_ani.append(C_pC_p)
        mu_f = mu_f1
        C_f = C_f1
        std = np.sqrt(C_pC_p)
        U = M(mumu = mu, sigsig = std, p = 0.1, q = 10)
        mu_min_iti = np.unravel_index(mu.argmin(), mu.shape)
        mu_min = np.amin(mu)
        semi.kl_score = mu_min
        semi.kl_x = datax[mu_min_iti[0]]
        semi.kl_y = datay[mu_min_iti[1]]
        semi.kl_z = dataz[mu_min_iti[2]]
        rospy.loginfo('mu_min: %f', mu_min)
        rospy.loginfo('datax[mu_min_iti[0]]: %f', datax[mu_min_iti[0]])
        rospy.loginfo('datay[mu_min_iti[1]]: %f', datay[mu_min_iti[1]])
        rospy.loginfo('dataz[mu_min_iti[2]]: %f', dataz[mu_min_iti[2]])
        # U_max_iti = np.unravel_index(U.argmax(), U.shape)
        # U_max = np.amax(U)
        # semi.kl_score = U_max
        # semi.kl_x = datax[U_max_iti[0]]
        # semi.kl_y = datay[U_max_iti[1]]
        # semi.kl_z = dataz[U_max_iti[2]]
        # rospy.loginfo('U_max: %f', U_max)
        # rospy.loginfo('datax[U_max_iti[0]]: %f', datax[U_max_iti[0]])
        # rospy.loginfo('datay[U_max_iti[1]]: %f', datay[U_max_iti[1]])
        # rospy.loginfo('dataz[U_max_iti[2]]: %f', dataz[U_max_iti[2]])
        # mu_max_iti = np.unravel_index(mu.argmax(), mu.shape)
        # mu_max = np.amax(mu)
        # # semi.kl_score = mu_max
        # # semi.kl_x = datax[mu_max_iti[0]]
        # # semi.kl_y = datay[mu_max_iti[1]]
        # # semi.kl_z = dataz[mu_max_iti[2]]
        # print('mu_max', mu_max)
        # # print('mu_max_iti[0]', mu_max_iti[0])
        # # print('mu_max_iti[1]', mu_max_iti[1])
        # # print('mu_max_iti[2]', mu_max_iti[2])
        # print('datax[mu_max_iti[0]]', datax[mu_max_iti[0]])
        # print('datay[mu_max_iti[1]]', datay[mu_max_iti[1]])
        # print('dataz[mu_max_iti[2]]', dataz[mu_max_iti[2]])
        timeout3 = time.time() - time_get
        print('ok2')
        # semi.saiteki_kl = saiteki_KL
        # semi.sinrai_kl = lb_kl_sigma
        # semi.jizen_heikin = mu_f
        # semi.jizen_bunsan = C_f
########################################
########################################
########################################
########################################
########################################
########################################
########################################
########################################
        
	
def gauss_generate():
    kl_py_pub = rospy.Publisher('semi_opt', kl_suzu, queue_size = 10)
    rospy.init_node('gauss_generate')
    kl_sub1 = rospy.Subscriber("kl_suzuki", kl_suzuki, klCallback1, queue_size=1)
    r = rospy.Rate(30)
    while not rospy.is_shutdown():
        kl_py_pub.publish(semi)
         #print(semi.kl_score)
        r.sleep()
        
    # rospy.spin()
if __name__ == '__main__':
    try:
        gauss_generate()
    except rospy.ROSInterruptException: pass