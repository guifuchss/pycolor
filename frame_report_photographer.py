import sys
import numpy as np
from PIL import Image, ImageDraw, ImageTk
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def process_image(image_path, save_path):
    width = 8.27  # inches
    height = 11.69  # inches

    img = Image.open(image_path)
    image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    img_rgb = img.convert('RGB')

    # Resize the images to have the same size
    img_resized = img.resize((int(img.width * 0.5), int(img.height * 0.5)))

    r_avg, g_avg, b_avg = [], [], []

    for column in range(img_resized.size[0]):
        r, g, b = 0, 0, 0
        for pixel in img_resized.crop((column, 0, column + 1, img_resized.size[1])).getdata():
            r += pixel[0]
            g += pixel[1]
            b += pixel[2]
        r_avg.append(r / img_resized.size[1])
        g_avg.append(g / img_resized.size[1])
        b_avg.append(b / img_resized.size[1])

    if not r_avg or not g_avg or not b_avg:
        print(f"Empty channel data for {image_path}. Skipping this image.")
        return

    max_val = max(max(r_avg), max(g_avg), max(b_avg))
    r_avg = [x / max_val * 100 for x in r_avg]
    g_avg = [x / max_val * 100 for x in g_avg]
    b_avg = [x / max_val * 100 for x in b_avg]

    x = np.arange(len(r_avg))

    img_array = np.array(img)

    hsv = cv2.cvtColor(np.array(img_resized), cv2.COLOR_RGB2HSV)
    value = hsv[:, :, 2]
    saturation = hsv[:, :, 1]

    shadow_threshold = 30
    highlight_threshold = 80

    shadows_mask = value < shadow_threshold
    midtones_mask = (value >= shadow_threshold) & (value < highlight_threshold)
    highlights_mask = value >= highlight_threshold

    fig = plt.figure(figsize=(width, height), facecolor='#2E2E2E')
    gs = gridspec.GridSpec(3, 1, height_ratios=[2, 2, 1])

    ax0 = plt.subplot(gs[0])
    ax0.imshow(img_resized)
    ax0.axis('off')

    IRErange = [-7, 2, 8, 15, 24, 43, 47, 54, 58, 77, 84, 93, 100, 109]
    colorBands = [[77,41,80],[19,102,148],[29,133,160],[71,164,168],[133,133,133],[102,183,77],[159,159,159],[247,129,117],[209,209,209],[240,229,145],[255,253,56],[253,140,37],[252,13,27]]

    def processImageWithLegend(grey):
        numColorBands = len(colorBands)

        out = Image.new('RGB', grey.size)
        for i in range(0,len(IRErange)-1):
            vmin = int(255/109.0*IRErange[i])
            vmax = int(255/109.0*IRErange[i+1])
            mask = grey.point(lambda i: i >= vmin and i < vmax and 255)
            out.paste(tuple(colorBands[i]),None,mask)

        return out

    grey = img_resized.convert('L')
    false_color_image = processImageWithLegend(grey)
    false_color_image = np.array(false_color_image)

    ax1 = plt.subplot(gs[1])
    ax1.imshow(false_color_image)
    ax1.axis('off')

    def draw_colorbar(ax, ticks, tick_labels):
        norm = plt.Normalize(0, 1)
        cmap = plt.cm.jet
        cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, orientation='vertical', fraction=0.02, pad=0.04)
        cb.set_ticks(ticks)
        cb.set_ticklabels(tick_labels)
        cb.ax.tick_params(labelsize=8, colors='white')
        cb.outline.set_edgecolor('white')

    ticks = [i / (len(IRErange) - 1) for i in range(len(IRErange))]
    tick_labels = [f'{IRErange[i]}' for i in range(len(IRErange))]
    draw_colorbar(ax1, ticks, tick_labels)

    def plot_waveform(ax, img):
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        y, _, _ = cv2.split(img_yuv)
        height, width = y.shape

        y = (y / 255.0) * 100

        ax.set_xlim(0, width)
        ax.set_ylim(0, 100)
        ax.set_facecolor('#2E2E2E')
        ax.grid(color='yellow', linestyle='-', linewidth=0.5, alpha=0.3)

        for i in range(width):
            y_column = y[:, i]
            x = np.full(y_column.shape, i)
            ax.scatter(x, y_column, color='white', s=0.1)

        ax.set_title('Waveform', color='white')
        ax.set_ylabel('Intensity (IRE)', color='white')
        ax.set_xticks([])
        ax.set_yticks(np.arange(0, 101, 10))
        ax.yaxis.set_tick_params(colors='white')
        ax.spines['left'].set_color('white')
        ax.spines['left'].set_linewidth(0.5)
        ax.spines['bottom'].set_color('white')
        ax.spines['bottom'].set_linewidth(0.5)
        ax.tick_params(axis='y', which='both', colors='white')

    ax2 = plt.subplot(gs[2], facecolor='#2E2E2E')
    plot_waveform(ax2, image)

    plt.subplots_adjust(hspace=0.3)
    plt.savefig(save_path, format='jpeg', dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python frame_report_photographer.py <image_path> <save_path>")
    else:
        image_path = sys.argv[1]
        save_path = sys.argv[2]
        process_image(image_path, save_path)
