import matplotlib as plt


def plot_img_mask_and_real_and_change(img1, mask, img2, img3, channel, loss0, loss1, F_measure=-1):
    fig = plt.figure()
    plt.title('all loss ={:.3f}, channel {} loss {}'.format(loss0, channel, loss1))
    a = fig.add_subplot(2, 2, 1)
    a.set_title('raw')
    plt.imshow(img1)

    b = fig.add_subplot(2, 2, 2)
    if F_measure != -1:
        b.set_title('F-measure = {:.3f}'.format(F_measure))
    else:
        b.set_title('Output')
    plt.imshow(mask)

    c = fig.add_subplot(2, 2, 3)
    c.set_title('output_sigmoid')
    plt.imshow(img2)

    a = fig.add_subplot(2, 2, 4)
    a.set_title('mask')
    plt.imshow(img3)

    num = len(os.listdir('./result/train_featuremap'))
    plt.savefig('./result/valuation_featuremap/predict_{}-all_loss_{:.3f}-loss{:.3f}.jpg'.format(num, loss0, loss1),
                dpi=600)

    plt.close()


def train_plot_img_mask_and_real_and_change(img1, mask, img2, img3, channel, loss0, loss1):
    fig = plt.figure()
    plt.title('all loss ={:.3f}, channel {} loss {}'.format(loss0, channel, loss1))
    a = fig.add_subplot(2, 2, 1)
    a.set_title('raw')
    plt.imshow(img1)

    b = fig.add_subplot(2, 2, 2)
    b.set_title('Output')
    plt.imshow(mask)

    c = fig.add_subplot(2, 2, 3)
    c.set_title('output_sigmoid')
    plt.imshow(img2)

    d = fig.add_subplot(2, 2, 4)
    d.set_title('mask')
    plt.imshow(img3)

    num = len(os.listdir('./result/train_featuremap'))
    plt.savefig('./result/train_featuremap/predict_{}-all_loss_{:.3f}-loss{:.3f}.jpg'.format(num, loss0, loss1),
                dpi=600)

    plt.close()
