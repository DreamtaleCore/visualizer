"""
In this script, combine image pair(input & output) together and yield a GIF for a better visualization.
Can be used as a teaser of project page.

::author: DreamTale dreamtalewind@gmail.com
"""
import os
import cv2
import imageio
import numpy as np

# region Pre-definition space
PWD_INPUT = 'data/input/_1_Focus_11_m.jpg'
PWD_TARGET = 'data/target/_1_Focus_11_g.jpg'

OUTPUT_CONFIG = {
    'mode': 'slide window',         # in ['slide bar', 'blend', 'slide window']
    'size': 512,                    # int, the new long edge of output image
    'iter': 31,                     # the number of GIF iterations
    'dir': 'data/out_img/teaser'    # output dir
}

SLIDE_BAR_CONFIG = {
    'color': (255, 0, 255),         # [blue, green, red] in (0, 255)
    'ori': 'vertical',              # in ['vertical', 'horizontal']
    'width': 5,                     # width of the slide bar
    'pattern pwd': None,            # None or the pattern image path
    'pad color': (255, 255, 255)    # padding color
}

SLIDE_WIN_CONFIG = {
    'pwd': 'data/residual/_1_Focus_11_r.jpg',   # The residual image's path
    'skipped width': 5,                         # The space width between the output images
    'ori': 'horizontal',                        # in ['vertical', 'horizontal']
    'pad color': (0, 0, 0),                     # padding color
}

if not os.path.exists(OUTPUT_CONFIG['dir']):
    os.makedirs(OUTPUT_CONFIG['dir'])
# endregion

# region Main work space
img_input = cv2.imread(PWD_INPUT)
img_target = cv2.imread(PWD_TARGET)
h, w, _ = img_input.shape
if h > w:
    new_h, new_w = OUTPUT_CONFIG['size'], int(w * OUTPUT_CONFIG['size'] / h)
else:
    new_h, new_w = int(h * OUTPUT_CONFIG['size'] / w), OUTPUT_CONFIG['size']

img_input = cv2.resize(img_input, (new_w, new_h))
img_target = cv2.resize(img_target, (new_w, new_h))
img_canvas = None
img_buf = []

print('Processing ... please do NOT exist this program.')
if OUTPUT_CONFIG['mode'] == 'slide bar':
    margin = SLIDE_BAR_CONFIG['width']

    img_canvas = np.zeros(shape=(new_h + 2 * margin, new_w + 2 * margin, 3), dtype=np.int32)
    img_canvas += SLIDE_BAR_CONFIG['pad color']

    # generate the slide bar
    if SLIDE_BAR_CONFIG['pattern pwd'] is not None:
        img_slide_bar = cv2.imread(SLIDE_BAR_CONFIG['pattern pwd'])
        if SLIDE_BAR_CONFIG['ori'] == 'vertical':
            if img_slide_bar.shape[0] < img_slide_bar.shape[1]:
                img_slide_bar = np.transpose(img_slide_bar, (1, 0, 2))
                img_slide_bar = np.flip(img_slide_bar, axis=0)
            img_slide_bar = cv2.resize(img_slide_bar, (margin, new_h + 2 * margin))
        else:
            if img_slide_bar.shape[0] > img_slide_bar.shape[1]:
                img_slide_bar = np.transpose(img_slide_bar, (1, 0, 2))
                img_slide_bar = np.flip(img_slide_bar, axis=0)
            img_slide_bar = cv2.resize(img_slide_bar, (new_w + 2 * margin, margin))
    else:
        if SLIDE_BAR_CONFIG['ori'] == 'vertical':
            _h_slider, _w_slider = new_h + 2 * margin, margin
        else:
            _h_slider, _w_slider = margin, new_w + 2 * margin
        img_slide_bar = np.zeros(shape=(_h_slider, _w_slider, 3), dtype=np.int32)
        img_slide_bar += SLIDE_BAR_CONFIG['color']

    for i in range(OUTPUT_CONFIG['iter']):
        if SLIDE_BAR_CONFIG['ori'] == 'vertical':
            slide_pos = int((new_w + margin) / (OUTPUT_CONFIG['iter'] - 1) * i)
            img_pos = slide_pos + margin // 2
            _img_split = np.clip(img_pos - margin, 0, new_w)
            _img_new = np.zeros(shape=[new_h, new_w, 3], dtype=np.int32)
            _img_new[:, : _img_split, :] += img_target[:, : _img_split, :]
            _img_new[:, _img_split:, :] += img_input[:, _img_split:, :]

            img_canvas = np.zeros(shape=(new_h + 2 * margin, new_w + 2 * margin, 3), dtype=np.int32)
            img_canvas += SLIDE_BAR_CONFIG['pad color']
            img_canvas[margin: margin + new_h, margin: margin + new_w, :] = _img_new
            _tmp_s = img_canvas[:, slide_pos: slide_pos + margin, :].shape
            img_canvas[:, slide_pos: slide_pos + margin, :] = img_slide_bar[:_tmp_s[0], :_tmp_s[1], :]
        else:
            slide_pos = int((new_h + margin) / (OUTPUT_CONFIG['iter'] - 1) * i)
            img_pos = slide_pos + margin // 2
            _img_split = np.clip(img_pos - margin, 0, new_h)
            _img_new = np.zeros(shape=[new_h, new_w, 3], dtype=np.int32)
            _img_new[: _img_split, :, :] += img_target[: _img_split, :, :]
            _img_new[_img_split:, :, :] += img_input[_img_split:, :, :]

            img_canvas = np.zeros(shape=(new_h + 2 * margin, new_w + 2 * margin, 3), dtype=np.int32)
            img_canvas += SLIDE_BAR_CONFIG['pad color']
            img_canvas[margin: margin + new_h, margin: margin + new_w, :] = _img_new
            _tmp_s = img_canvas[slide_pos: slide_pos + margin, :, :].shape
            img_canvas[slide_pos: slide_pos + margin, :, :] = img_slide_bar[:_tmp_s[0], :_tmp_s[1], :]
            pass

        # vis for debug
        img_canvas = img_canvas.astype(np.uint8)
        cv2.imshow('new image', img_canvas)
        cv2.waitKey(30)

        img_buf.append(img_canvas[:, :, ::-1])
        pass
    pass
elif OUTPUT_CONFIG['mode'] == 'blend':
    img_canvas = None
    for i in range(OUTPUT_CONFIG['iter']):
        alpha = i * 1 / OUTPUT_CONFIG['iter']
        img_canvas = cv2.addWeighted(img_target, alpha, img_input, (1 - alpha), gamma=0)

        # visualize for a better debug
        cv2.imshow('img', img_canvas)
        img_buf.append(img_canvas[:, :, ::-1])
        cv2.waitKey(30)

    for i in range(10):
        img_buf.append(img_canvas[:, :, ::-1])
elif OUTPUT_CONFIG['mode'] == 'slide window':
    img_residual = cv2.imread(SLIDE_WIN_CONFIG['pwd'])
    img_residual = cv2.resize(img_residual, (new_w, new_h))

    margin = SLIDE_WIN_CONFIG['skipped width']
    if SLIDE_WIN_CONFIG['ori'] == 'horizontal':
        img_canvas = np.zeros(shape=(new_h, new_w * 2 + margin, 3), dtype=np.int32)
        img_canvas = (img_canvas + SLIDE_WIN_CONFIG['pad color']).astype(np.uint8)
        img_blend = img_input.copy()
        pre_blend_step = OUTPUT_CONFIG['iter'] // 3
        for i in range(pre_blend_step):
            alpha = i * 1 / OUTPUT_CONFIG['iter']
            img = cv2.addWeighted(img_target, alpha, img_blend, 1 - alpha, gamma=0)

            img_canvas[:, :new_w, :] = img
            # visualize for a better debug
            cv2.imshow('img', img_canvas)
            img_buf.append(img_canvas[:, :, ::-1])
            cv2.waitKey(30)

        post_blend_step = OUTPUT_CONFIG['iter'] - pre_blend_step
        for i in range(post_blend_step):
            slide_pos = int((new_w + margin) / (post_blend_step - 1) * i)
            if slide_pos == 0:
                continue
            if slide_pos < new_w:
                _img_target = img_target[:, :slide_pos, :]
                _img_blend = (img_target[:, slide_pos:, :] * 0.7 +
                              img_residual[:, :new_w - slide_pos, :]).astype(np.uint8)
                _img_residual = img_residual[:, new_w - slide_pos:, :]
                _img_blank = np.zeros(shape=(new_h, new_w + margin - slide_pos, 3), dtype=np.int32)
                _img_blank = (_img_blank + SLIDE_WIN_CONFIG['pad color']).astype(np.uint8)
                _img_residual = cv2.hconcat([_img_residual, _img_blank])
            else:
                _img_target = img_target
                _img_blend = np.zeros(shape=(new_h, slide_pos - new_w, 3), dtype=np.int32)
                _img_blend = (_img_blend + SLIDE_WIN_CONFIG['pad color']).astype(np.uint8)
                _img_residual = img_residual

            img_canvas = cv2.hconcat([_img_target, _img_blend, _img_residual])
            cv2.imshow('img', img_canvas)
            img_buf.append(img_canvas[:, :, ::-1])
            cv2.waitKey(30)
    else:
        img_canvas = np.zeros(shape=(new_h * 2 + margin, new_w, 3), dtype=np.int32)
        img_canvas = (img_canvas + SLIDE_WIN_CONFIG['pad color']).astype(np.uint8)
        img_blend = img_input.copy()
        pre_blend_step = OUTPUT_CONFIG['iter'] // 3
        for i in range(pre_blend_step):
            alpha = i * 1 / OUTPUT_CONFIG['iter']
            img = cv2.addWeighted(img_target, alpha, img_blend, 1 - alpha, gamma=0)

            img_canvas[:new_h, :, :] = img
            # visualize for a better debug
            cv2.imshow('img', img_canvas)
            img_buf.append(img_canvas[:, :, ::-1])
            cv2.waitKey(30)

        post_blend_step = OUTPUT_CONFIG['iter'] - pre_blend_step
        for i in range(post_blend_step):
            slide_pos = int((new_h + margin) / (post_blend_step - 1) * i)
            if slide_pos == 0:
                continue
            if slide_pos < new_h:
                _img_target = img_target[:slide_pos, :, :]
                _img_blend = (img_target[slide_pos:, :, :] * 0.7 +
                              img_residual[:new_h - slide_pos, :, :]).astype(np.uint8)
                _img_residual = img_residual[new_h - slide_pos:, :, :]
                _img_blank = np.zeros(shape=(new_h + margin - slide_pos, new_w, 3), dtype=np.int32)
                _img_blank = (_img_blank + SLIDE_WIN_CONFIG['pad color']).astype(np.uint8)
                _img_residual = cv2.vconcat([_img_residual, _img_blank])
            else:
                _img_target = img_target
                _img_blend = np.zeros(shape=(slide_pos - new_h, new_w, 3), dtype=np.int32)
                _img_blend = (_img_blend + SLIDE_WIN_CONFIG['pad color']).astype(np.uint8)
                _img_residual = img_residual

            img_canvas = cv2.vconcat([_img_target, _img_blend, _img_residual])
            cv2.imshow('img', img_canvas)
            img_buf.append(img_canvas[:, :, ::-1])
            cv2.waitKey(30)
else:
    raise NotImplementedError
# endregion

for i in range(10):
    img_buf.append(img_canvas[:, :, ::-1])

gif = imageio.mimsave(os.path.join(OUTPUT_CONFIG['dir'], PWD_INPUT.split('/')[-1].split('.')[0] + '.gif'),
                      img_buf, 'GIF', duration=0.05)
print('Done! Press `Esc` to quit.')
if cv2.waitKey(0) == ord('q'):
    cv2.destroyAllWindows()
