from Librarys import *

from LoadImage import get_data, show_batch


from utils import Myconfig
from utils import show_image
from NNmodel import CNN




def train(model, train_loader, mc, criterion, epoch):
    since = time.time()
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    num_batches = 0
    avg_loss = 0
    running_corrects = 0
    with open('logs_new.txt', 'a') as file:
        for batch_idx, sample_batched in enumerate(train_loader):
            image, classes = sample_batched
            image, classes = Variable(image.type(mc.imgtype)), Variable(classes.type(mc.tartype))
            optimizer.zero_grad()
            # output = model(image)[0]
            output = model(image)
            _, preds = torch.max(output, 1)
            #print(output)
            # output = (output > 0.5).type(opt.dtype)	# use more gpu memory, also, loss does not change if use this line
            loss = criterion(output, classes)
            loss.backward()
            optimizer.step()
            running_corrects += torch.sum(preds == classes.data)
            # print(running_corrects)
            avg_loss += loss.data.item()
            num_batches += 1
        avg_loss /= num_batches
        avg_acc = running_corrects.double() / (num_batches * mc.batch_size)
        avg_acc = avg_acc.cpu().numpy()
        # accuracy = sum(output == classes['label'].cuda()) / float(classes['label'].size(0))
        time_elapsed = time.time() - since
        # avg_loss /= len(train_loader.dataset)
        print('epoch: ' + str(epoch) + ' train loss: ' + str(avg_loss) + ' accuracy: ' + str(avg_acc))
        print('Cost time: ', time_elapsed)
        # file.write('epoch: ' + str(epoch) + ' train loss: ' + str(avg_loss) + '\n')

        # tmpdf = pd.DataFrame({
        #     'epoch': [epoch],
        #     'loss': [avg_loss],
        #     'accuracy': [avg_acc]
        # })
        # tmpdf = tmpdf.append(tmpdf)
        # tmpdf.to_csv('information.csv')

        with open('train_loss_new.txt', 'a') as f1:
            f1.write(str(epoch) + ' ' + str(avg_loss) + ' ' + str(avg_acc) + '\n')

        file.write('epoch: ' + str(epoch) + ' ' + 'loss: ' + str(avg_loss) + '\n')


def val(model, val_loader, mc, criterion, epoch):
    model.eval()
    num_batches = 0
    avg_loss = 0
    running_corrects = 0



    with open('logs_new.txt', 'a') as file:
        for batch_idx, sample_batched in enumerate(val_loader):
            image, classes = sample_batched
            image, classes = Variable(image.type(mc.imgtype)), Variable(classes.type(mc.tartype))
            # output = model.forward(image)[0]
            output = model.forward(image)
            _, preds = torch.max(output, 1)
            # output = (output > 0.5).type(opt.dtype)	# use more gpu memory, also, loss does not change if use this line
            loss = criterion(output, classes)
            running_corrects += torch.sum(preds == classes.data)
            avg_loss += loss.data.item()
            num_batches += 1
        avg_loss /= num_batches
        avg_acc = running_corrects.double() / (num_batches * mc.batch_size)
        # avg_loss /= len(val_loader.dataset)

        print('epoch: ' + str(epoch) + ' validation loss: ' + str(avg_loss) + ' accuracy: ' + str(avg_acc.cpu().numpy()))
        # print('epoch: ' + str(epoch) + ', validation loss: ' + str(avg_loss))
        file.write('epoch: ' + str(epoch) + ', validation loss: ' + str(avg_loss) + '\n')

        return avg_acc


# train and validation
def run(model, train_loader, val_loader, mc, criterion, train_only):
    if train_only:
        for epoch in range(1, mc.epochs):
            model.load_state_dict(torch.load('normal.pt'))
            # model.load_state_dict(torch.load(os.path.join(mc.checkpoint_dir, 'model-01.pt')))
            train(model, train_loader, mc, criterion, epoch)
    else:

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        for epoch in range(1, mc.epochs):
            train(model, train_loader, mc, criterion, epoch)
            epoch_acc = val(model, val_loader, mc, criterion, epoch)

            # deep copy the model
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            print('\n\nbest_acc is: {}\n\n'.format(best_acc))
        # SAVE model
        if mc.save_model:
            print("Save model.....")
            torch.save(best_model_wts, os.path.join(mc.checkpoint_dir, 'model-resnet100_new.pt'))
            print("Save finished.....")


# make prediction
def run_test(model, test_loader, len_test, mc):
    """
    predict the masks on testing set
    :param model: trained model
    :param test_loader: testing set
    :param opt: configurations
    :return:
        - predictions: list, for each elements, numpy array (Width, Height)
        - img_ids: list, for each elements, an image id string
    """
    # predictions = []
    running_corrects = 0
    # print(model)
    wrong_detection_image = []
    for batch_idx, sample_batched in enumerate(test_loader):
        image, classes = sample_batched
        print(image, classes)
        image, classes = Variable(image.type(mc.imgtype)), Variable(classes.type(mc.tartype))
        output = model.forward(image)
        # output = (output > 0.5)
        # output = output.data.cpu().numpy()
        _, predicted = torch.max(output, 1)
        # print(predicted)
        # print(classes.data)
        running_corrects += torch.sum(predicted == classes.data)
        tmp = predicted - classes.data
        index = tmp.nonzero()
        for i in index:
            wrong_detection_image.append(image[i].cpu().numpy().squeeze())
        # print(running_corrects)
        # predictions.append([predicted])
    print('correct images: {}'.format(running_corrects.cpu().numpy()))
    # print(len_test)
    print('Accuracy is {}'.format(running_corrects.cpu().numpy()/len_test))

    return wrong_detection_image


if __name__ == '__main__':
    mc = Myconfig()
    model = CNN()

    if mc.is_train:
        # split all data to train and validation, set split = True
        train_loader, val_loader, class_names, device, len_data = get_data(mc.dataset_dir, True, mc)

        if mc.n_gpu > 1:
            model = nn.DataParallel(model)
        if mc.is_cuda:
            model = model.cuda()
        # optimizer = optim.Adam(model.parameters(), lr=mc.learning_rate, weight_decay=mc.weight_decay)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        # criterion = nn.BCELoss().cuda()
        criterion = nn.CrossEntropyLoss()

        # start to run a training
        run(model, train_loader, val_loader, mc, criterion, False)

        # make prediction on validation set
        # predictions = run_test(model, val_loader, len_test)

    elif mc.continue_train:
        model.load_state_dict(torch.load(os.path.join(mc.checkpoint_dir, 'model-01.pt')))
        train_loader, val_loader, class_names, device = get_data(mc.dataset_dir, True)
        if mc.n_gpu > 1:
            model = nn.DataParallel(model)
        if mc.is_cuda:
            model = model.cuda()
        # optimizer = optim.Adam(model.parameters(), lr=mc.learning_rate, weight_decay=mc.weight_decay)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        # criterion = nn.BCELoss().cuda()
        criterion = nn.CrossEntropyLoss()
        # start to run a training
        run(model, train_loader, val_loader, mc, criterion, False)

        # make prediction on validation set
        predictions = run_test(model, val_loader, mc)

        # SAVE model
        if mc.save_model:
            print("Save model.....")
            torch.save(model.state_dict(), os.path.join(mc.checkpoint_dir, 'model-01.pt'))
            print("Save finished.....")
    else:
        # load testing data for making predictions
        test_loader, class_names, device, len_test = get_data(mc.testset_dir, False)
        # load the model and run test
        model.load_state_dict(torch.load(os.path.join(mc.checkpoint_dir, 'model-01.pt')))
        if mc.n_gpu > 1:
            model = nn.DataParallel(model)
        if mc.is_cuda:
            model = model.cuda()
        wrong_detection_image = run_test(model, test_loader, len_test)
        # wrong_detection_image = np.array(wrong_detection_image)
        # print(wrong_detection_image[0].shape)
        show_image(wrong_detection_image)



