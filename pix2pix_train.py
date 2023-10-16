import torch
from torchvision import transforms
from torch.autograd import Variable
from dataset import DatasetFromFolder
from model import Generator, Generator128, Discriminator, Discriminator128
import utils
import argparse
import os
from logger import Logger
import frame_dataset


def load_pretrained_model(G, D, model_save_path, pretrained_model):
    G.load_state_dict(torch.load(os.path.join(
        model_save_path, 'generator_param_{}.pkl'.format(pretrained_model))))
    D.load_state_dict(torch.load(os.path.join(
        model_save_path, 'discriminator_param_{}.pkl'.format(pretrained_model))))

    print("Restored model (epoch: {})".format(pretrained_model))

    return G, D

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=False, default='facades', help='input dataset')
parser.add_argument('--direction', required=False, default='BtoA', help='input and target image order')
parser.add_argument('--batch_size', type=int, default=4, help='train batch size')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--input_size', type=int, default=256, help='input size')
parser.add_argument('--resize_scale', type=int, default=286, help='resize scale (0 is false)')
parser.add_argument('--crop_size', type=int, default=256, help='crop size (0 is false)')
parser.add_argument('--fliplr', type=bool, default=True, help='random fliplr True of False')
parser.add_argument('--num_epochs', type=int, default=15, help='number of train epochs')
parser.add_argument('--lrG', type=float, default=0.0002, help='learning rate for generator, default=0.0002')
parser.add_argument('--lrD', type=float, default=0.0002, help='learning rate for discriminator, default=0.0002')
parser.add_argument('--lamb', type=float, default=100, help='lambda for L1 loss')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
parser.add_argument('--pretrained_model', type=str, default=None, help='model number to continue training existing model')
params = parser.parse_args()
print(params)

# Directories for loading data and saving results
data_dir = '../data/' + params.dataset + '/'
save_dir = params.dataset + '_results/'
model_dir = params.dataset + '_model/'

if not os.path.exists(save_dir):
    os.mkdir(save_dir)
if not os.path.exists(model_dir):
    os.mkdir(model_dir)

# Data pre-processing
transform = transforms.Compose([transforms.Resize(params.input_size),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])


if params.dataset == 'frames':
    train_data_loader = torch.utils.data.DataLoader(frame_dataset.FrameDataset(params.crop_size, data_dir),
                                                batch_size=params.batch_size,
                                                shuffle=True)
    test_data_loader = torch.utils.data.DataLoader(frame_dataset.FrameDataset(params.crop_size, '../data/test'),
                                                batch_size=params.batch_size,
                                                shuffle=True)
else:
    # Train data
    train_data = DatasetFromFolder(data_dir, subfolder='train', direction=params.direction, transform=transform,
                                   resize_scale=params.resize_scale, crop_size=params.crop_size, fliplr=params.fliplr)
    train_data_loader = torch.utils.data.DataLoader(dataset=train_data,
                                                    batch_size=params.batch_size,
                                                    shuffle=True)

    # Test data
    test_data = DatasetFromFolder(data_dir, subfolder='test', direction=params.direction, transform=transform)
    test_data_loader = torch.utils.data.DataLoader(dataset=test_data,
                                                   batch_size=params.batch_size,
                                                   shuffle=False)
test_input, test_target = test_data_loader.__iter__().__next__()

# Models
if params.dataset == 'frames':
    G = Generator(6, params.ngf, 3, True)
    D = Discriminator(9, params.ndf, 1, True)
else:
    G = Generator(3, params.ngf, 3)
    D = Discriminator(6, params.ndf, 1)
G.cuda()
D.cuda()
G.normal_weight_init(mean=0.0, std=0.02)
D.normal_weight_init(mean=0.0, std=0.02)

if params.pretrained_model:
    G,D = load_pretrained_model(G, D, model_dir, params.pretrained_model)

# Set the logger
D_log_dir = save_dir + 'D_logs'
G_log_dir = save_dir + 'G_logs'
if not os.path.exists(D_log_dir):
    os.mkdir(D_log_dir)
D_logger = Logger(D_log_dir)

if not os.path.exists(G_log_dir):
    os.mkdir(G_log_dir)
G_logger = Logger(G_log_dir)

# Loss function
BCE_loss = torch.nn.BCELoss().cuda()
L1_loss = torch.nn.L1Loss().cuda()

# Optimizers
G_optimizer = torch.optim.Adam(G.parameters(), lr=params.lrG, betas=(params.beta1, params.beta2))
D_optimizer = torch.optim.Adam(D.parameters(), lr=params.lrD, betas=(params.beta1, params.beta2))

# Training GAN
D_avg_losses = []
G_avg_losses = []

if params.pretrained_model:
    start = int(params.pretrained_model)
else:
    start = 0

step = 0
for epoch in range(start, params.num_epochs):
    D_losses = []
    G_losses = []

    # training
    for i, (input, target) in enumerate(train_data_loader):

        # input & target image data
        if params.dataset == 'frames':
            x_ = [Variable(input[0].cuda()), Variable(input[1].cuda())]
        else:
            x_ = Variable(input.cuda())
        y_ = Variable(target.cuda())

        # Train discriminator with real data
        D_real_decision = D(x_, y_).squeeze()
        real_ = Variable(torch.ones(D_real_decision.size()).cuda())
        D_real_loss = BCE_loss(D_real_decision, real_)

        # Train discriminator with fake data
        gen_image = G(x_)
        D_fake_decision = D(x_, gen_image).squeeze()
        fake_ = Variable(torch.zeros(D_fake_decision.size()).cuda())
        D_fake_loss = BCE_loss(D_fake_decision, fake_)

        # Back propagation
        D_loss = (D_real_loss + D_fake_loss) * 0.5
        D.zero_grad()
        D_loss.backward()
        D_optimizer.step()

        # Train generator
        gen_image = G(x_)
        D_fake_decision = D(x_, gen_image).squeeze()
        G_fake_loss = BCE_loss(D_fake_decision, real_)

        # L1 loss
        l1_loss = params.lamb * L1_loss(gen_image, y_)

        # Back propagation
        G_loss = G_fake_loss + l1_loss
        G.zero_grad()
        G_loss.backward()
        G_optimizer.step()

        # loss values
        D_losses.append(D_loss.data.item())
        G_losses.append(G_loss.data.item())

        if i % 100 == 0:
            print('Epoch [%d/%d], Step [%d/%d], D_loss: %.4f, G_loss: %.4f'
                  % (epoch+1, params.num_epochs, i+1, len(train_data_loader), D_loss.data.item(), G_loss.data.item()))

        # ============ TensorBoard logging ============#
        D_logger.scalar_summary('discriminator', D_loss.data.item(), step)
        G_logger.scalar_summary('generator', G_loss.data.item(), step)
        step += 1
        

    D_avg_loss = torch.mean(torch.FloatTensor(D_losses))
    G_avg_loss = torch.mean(torch.FloatTensor(G_losses))

    # avg loss values for plot
    D_avg_losses.append(D_avg_loss)
    G_avg_losses.append(G_avg_loss)

    # Show result for test image
    if params.dataset == 'frames':
        test_input = [Variable(test_input[0].cuda()), Variable(test_input[1].cuda())]
        gen_image = G(test_input)
        gen_image = gen_image.cpu().data
        test_input = [Variable(test_input[0].cpu()), Variable(test_input[1].cpu())]
        utils.plot_test_result(test_input, test_target, gen_image, epoch, save=True, save_dir=save_dir, is_frames=True)
    else:
        gen_image = G(Variable(test_input.cuda()))
        gen_image = gen_image.cpu().data
        utils.plot_test_result(test_input, test_target, gen_image, epoch, save=True, save_dir=save_dir)
        
    torch.save(G.state_dict(), model_dir + 'generator_param_'+str(epoch)+'.pkl')
    torch.save(D.state_dict(), model_dir + 'discriminator_param_'+str(epoch)+'.pkl')
    

# Plot average losses
utils.plot_loss(D_avg_losses, G_avg_losses, params.num_epochs, save=True, save_dir=save_dir)

# Make gif
utils.make_gif(params.dataset, params.num_epochs, save_dir=save_dir)

# Save trained parameters of model
torch.save(G.state_dict(), model_dir + 'generator_param.pkl')
torch.save(D.state_dict(), model_dir + 'discriminator_param.pkl')
