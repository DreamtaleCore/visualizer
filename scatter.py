"""
In this script, visualize different data-pair into a 2-dimension scatter map for a better visualization.
Can be used to visualize the numerical comparison among different methods

::author: DreamTale dreamtalewind@gmail.com
"""
import matplotlib.pyplot as plt
import collections
import os

# region Pre-definition space
MethodData = collections.namedtuple('MethodData', ('name', 'ssim', 'psnr', 'time', 'pattern', 'offset'))
method_list = [
    MethodData(name='Input', ssim=0.801, psnr=19.024, time=0.0, pattern='D', offset=(0, 0)),
    MethodData(name='AN17', ssim=0.786, psnr=19.227, time=99.349, pattern='^', offset=(-58, -5)),
    MethodData(name='LB14', ssim=0.763, psnr=17.772, time=0.475, pattern='>', offset=(-23, 10)),
    MethodData(name='FY17', ssim=0.810, psnr=20.946, time=0.0951, pattern='s', offset=(-22, 10)),
    MethodData(name='YG18', ssim=0.800, psnr=20.030, time=0.0244, pattern='p', offset=(-20, -25)),
    MethodData(name='WS18', ssim=0.812, psnr=19.030, time=0.619, pattern='v', offset=(-20, 10)),
    MethodData(name='Ours', ssim=0.860, psnr=23.093, time=0.0612, pattern='*', offset=(-25, -20)),
    MethodData(name='ZN18', ssim=0.846, psnr=21.364, time=0.332, pattern='d', offset=(9, -5)),
]

font_figure_size = 20
font_axis = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 34,
             }
img_post_fix = '.png'

fig_size = (7, 6)

output_dir = 'data/out_img/scatter'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
# endregion

# region Main work space
plt.figure(figsize=fig_size)
plt.xlabel('Test time (sec.)', font_axis)
plt.ylabel('SSIM', font_axis)
for item in method_list:
    if 'ours' in item.name.lower():
        plt.scatter(item.time, item.ssim, marker=item.pattern, s=240)
        plt.annotate(item.name, xy=(item.time, item.ssim), xycoords='data', xytext=item.offset,
                     textcoords='offset points', fontsize=font_figure_size, fontweight='semibold', color='red')
    else:
        plt.scatter(item.time, item.ssim, marker=item.pattern, s=200)
        plt.annotate(item.name, xy=(item.time, item.ssim), xycoords='data', xytext=item.offset,
                     textcoords='offset points', fontsize=font_figure_size)

plt.xscale('log')
plt.savefig(os.path.join(output_dir, 'ssim_vs_time') + img_post_fix)
plt.show()

plt.figure(figsize=fig_size)
plt.xlabel('Test time (sec.)', font_axis)
plt.ylabel('PSNR', font_axis)
for item in method_list:
    if 'ours' in item.name.lower():
        plt.scatter(item.time, item.psnr, marker=item.pattern, s=120)
        plt.annotate(item.name, xy=(item.time, item.psnr), xycoords='data', xytext=item.offset,
                     textcoords='offset points', fontsize=font_figure_size, fontweight='semibold', color='red')
    else:
        plt.scatter(item.time, item.psnr, marker=item.pattern, s=100)
        plt.annotate(item.name, xy=(item.time, item.psnr), xycoords='data', xytext=item.offset,
                     textcoords='offset points', fontsize=font_figure_size)

plt.xscale('log')
plt.savefig(os.path.join(output_dir, 'psnr_vs_time') + img_post_fix)
plt.show()
# endregion
