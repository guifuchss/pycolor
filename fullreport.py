import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import matplotlib.gridspec as gridspec
import tkinter as tk
from tkinter import simpledialog
from tkinter import filedialog
from matplotlib.backends.backend_pdf import PdfPages
import sys
import tkinter.messagebox as messagebox
from matplotlib import colors as mcolors
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter



def show_skip_dialog(title, prompts):
    root = tk.Tk()
    root.withdraw()

    top = tk.Toplevel(root)
    top.title(title)

    results = []

    def submit():
        names = [entry.get() for entry in name_entries]
        results.extend(names)
        top.destroy()

    def skip():
        results.append('-')
        top.destroy()

    name_frames = []
    name_entries = []

    for prompt in prompts:
        frame = tk.Frame(top)
        label = tk.Label(frame, text=prompt)
        label.pack(side=tk.LEFT, padx=10)

        entry = tk.Entry(frame)
        entry.pack(side=tk.LEFT)
        name_entries.append(entry)

        frame.pack(pady=10)
        name_frames.append(frame)

    button_frame = tk.Frame(top)


    submit_button = tk.Button(button_frame, text='Submit', command=submit)
    submit_button.pack(side=tk.LEFT)

    button_frame.pack(pady=10)

    top.wait_window()

    return results



# Prompt the user to enter the information
prompts = [
    "Filme:",
    "Diretor:",
    "Colorista:",
    "Diretor de Arte:",
    "Diretor de Fotografia:"
]

# Access the names entered by the user
names = show_skip_dialog(title="Informações", prompts=prompts)

# Access the names entered by the user
movie_name, director_name, colorist_name, art_director_name, dp_name = names


if len(sys.argv) > 1:
    image_path = sys.argv[1]
else:
    image_path = filedialog.askopenfilename(title="Select Image")

# Abrir caixa de diálogo para salvar arquivo
file_path = filedialog.asksaveasfilename(defaultextension='.pdf')

# Inicializar o objeto PdfPages
pp = PdfPages(file_path)

width = 8.27  # inches
height = 11.69  # inches

# Read Image
img = Image.open(image_path)
image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def cluster_image(image, n_clusters):
    # Remove all black (also shades close to black) pixels before clustering
    image = image[(image > [10, 10, 10]).all(axis=2)]

    pixels = image.reshape(-1, 3)
    kmeans = KMeans(n_clusters=n_clusters, n_init=10)
    labels = kmeans.fit_predict(pixels)

    # Count the occurrence of each label to determine the predominance
    counter = np.bincount(labels)
    total_pixels = sum(counter)
    count_labels = list(range(len(counter)))

    # Sort the clusters by frequency (from less to more frequent)
    count_labels.sort(key=lambda x: counter[x])

    # Compute the width of each color in the palette
    widths = [round(500 * count / total_pixels) for count in counter]

    # Adjust the widths so that their sum is exactly 500
    while sum(widths) < 500:
        widths[widths.index(min(widths))] += 1
    while sum(widths) > 500:
        widths[widths.index(max(widths))] -= 1

    palette = np.zeros([50, 500, 3], dtype=np.uint8)

    current_x = 0
    for i in range(n_clusters):
        next_x = current_x + widths[count_labels[i]]
        palette[:, current_x:next_x, :] = kmeans.cluster_centers_[count_labels[i]].astype(int)
        current_x = next_x

    return palette

def plot_image(image, gs, title):
    # Convert the image from BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Now use Matplotlib's imshow() function to display the image
    ax = plt.subplot(gs)
    ax.imshow(image_rgb)
    ax.set_title(title, color='white', fontsize=20)
    ax.axis('off')


def create_palette(img, n_colors):
    # Convert image data into 2D array
    data = np.array(img).reshape((-1, 3))

    # Apply k-means to find most dominant colors
    kmeans = KMeans(n_clusters=n_colors, n_init=10)
    kmeans.fit(data)

    # Get colors
    colors = kmeans.cluster_centers_

    # Ensure values are within 0-255 range
    colors = colors.round(0).astype(int)

    # Get the count of pixels in each cluster
    unique, counts = np.unique(kmeans.labels_, return_counts=True)
    dominant_colors = dict(zip(unique, counts))

    # Sort colors by dominance
    sorted_colors = sorted(dominant_colors.items(), key=lambda item: item[1], reverse=True)

    # Prepare colors for plotting
    plot_colors = [colors[i] for i, count in sorted_colors]

    # Normalize counts for plotting
    counts = [count for i, count in sorted_colors]
    counts = np.array(counts) / sum(counts)

    return plot_colors, counts
def extract_dominant_color(image_segment):
    # Convert image data into 2D array
    data = np.array(image_segment).reshape((-1, 3))

    # Apply k-means to find most dominant color
    kmeans = KMeans(n_clusters=1, n_init=10)
    kmeans.fit(data)

    # Get the dominant color
    dominant_color = kmeans.cluster_centers_[0]

    # Ensure values are within 0-255 range
    dominant_color = dominant_color.round(0).astype(int)

    return dominant_color / 255  # normalize color

def plot_colors(colors, counts):
    fig, ax = plt.subplots(1, 1, figsize=(10.40, 2),
                            dpi=80, facecolor='#2E2E2E', edgecolor='k')

# Call the functions
colors, counts = create_palette(img, 10)

# Convert the image to RGB
img_rgb = img.convert('RGB')

# Create a numpy array of floats to store the average values obtained by
# averaging the pixels along the width of the image
r_avg = []
g_avg = []
b_avg = []

# Loop over the image data
for column in range(img_rgb.size[0]):
    # Get a list of all pixels in a column
    r, g, b = 0, 0, 0
    for pixel in img_rgb.crop((column, 0, column + 1, img_rgb.size[1])).getdata():
        r += pixel[0]
        g += pixel[1]
        b += pixel[2]
    # Append the average for each channel to respective list
    r_avg.append(r / img_rgb.size[1])
    g_avg.append(g / img_rgb.size[1])
    b_avg.append(b / img_rgb.size[1])

# Normalizing average values
max_val = max(max(r_avg), max(g_avg), max(b_avg))
r_avg = [x / max_val * 100 for x in r_avg]
g_avg = [x / max_val * 100 for x in g_avg]
b_avg = [x / max_val * 100 for x in b_avg]

# Create the x locations for the groups
x = np.arange(len(r_avg))

# Plot the RGB histogram
img_array = np.array(img)

# Convert the image to HSV
hsv = cv2.cvtColor(np.array(img_rgb), cv2.COLOR_RGB2HSV)

# Extract the value and saturation channels
value = hsv[:, :, 2]
saturation = hsv[:, :, 1]

# Define the thresholds for shadows, midtones, and highlights
shadow_threshold = 30
highlight_threshold = 80

# Create masks for shadows, midtones, and highlights
shadows_mask = value < shadow_threshold
midtones_mask = (value >= shadow_threshold) & (value < highlight_threshold)
highlights_mask = value >= highlight_threshold

# Calculate the average saturation for shadows, midtones, and highlights
average_saturation_shadows = np.mean(saturation[shadows_mask])
average_saturation_midtones = np.mean(saturation[midtones_mask])
average_saturation_highlights = np.mean(saturation[highlights_mask])

# Create a list with the average saturations
average_saturations = [average_saturation_shadows, average_saturation_midtones, average_saturation_highlights]

shadows_color = extract_dominant_color(hsv[shadows_mask])
midtones_color = extract_dominant_color(hsv[midtones_mask])
highlights_color = extract_dominant_color(hsv[highlights_mask])

shadows_color_rgb = mcolors.hsv_to_rgb(shadows_color)
midtones_color_rgb = mcolors.hsv_to_rgb(midtones_color)
highlights_color_rgb = mcolors.hsv_to_rgb(highlights_color)

bar_colors = [shadows_color_rgb, midtones_color_rgb, highlights_color_rgb]


# Create a list with the names of the segments
segments = ['Sombras', 'Meio-Tons', 'Altas-Luzes']

fig = plt.figure(figsize=(width, height), facecolor='#2E2E2E')
gs = gridspec.GridSpec(5, 1, height_ratios=[3, 1, 1, 1, 1])

plt.figtext(0.5, 0.98, f' {movie_name}', fontsize=14, fontweight='bold', color='white', ha='center')  # Movie name (in bold)
plt.figtext(0.5, 0.96, f'Diretor: {director_name}', fontsize=7, color='white', ha='center')  # Director's name
plt.figtext(0.5, 0.94, f'Diretor de Fotografia: {dp_name}', fontsize=7, color='white', ha='center')  # Director of Photography's name
plt.figtext(0.5, 0.92, f'Diretor de Arte: {art_director_name}', fontsize=7, color='white', ha='center')  # Art Director's name
plt.figtext(0.5, 0.90, f'Colorista: {colorist_name}', fontsize=7, color='white', ha='center')  # Colorist's name






# Plot Image
ax0 = plt.subplot(gs[0])
ax0.imshow(img)
ax0.axis('off')

# Call the functions
colors, counts = create_palette(img, 10)

# Normalize colors
colors = [(r / 255, g / 255, b / 255) for r, g, b in colors]

# Color bar plot
ax1 = plt.subplot(gs[1])
start = 0
for i in range(len(colors)):
    ax1.bar(start, 1, width=counts[i], color=colors[i], align='edge', edgecolor='none')
    start += counts[i]

ax1.set_xlim([0, 1])
ax1.set_ylim([0, 1])
ax1.set_aspect('auto')
ax1.axis('off')

# Plot the 'waveform'
ax2 = plt.subplot(gs[2], facecolor='#2E2E2E')
ax2.plot(x, r_avg, 'r', x, g_avg, 'g', x, b_avg, 'b')
ax2.set_xlim(left=0)
ax2.set_ylim(bottom=0)
ax2.set_title('Waveform', color='white')

# Remover rótulos e marcações do eixo x
ax2.set_xticks([])  # Remove as marcações do eixo x
ax2.set_xticklabels([])  # Remove os rótulos do eixo x

ax2.tick_params(axis='y', colors='white')


# Plot the RGB histogram
ax3 = plt.subplot(gs[3], facecolor='#2E2E2E')
for i, color in enumerate(['r', 'g', 'b']):
    ax3.hist(img_array[..., i].ravel(), bins=256, color=color, alpha=0.5)
ax3.set_xlim(left=0)
ax3.set_ylim(bottom=0)
ax3.set_title('Histograma', color='white')
ax3.tick_params(axis='x', colors='white')
ax3.tick_params(axis='y', colors='white')
ax3.set_yticks([])  # Remove as marcações do eixo x
ax3.set_yticklabels([])  # Remove os rótulos do eixo x

plt.subplots_adjust(hspace=0.5)

# Criar um gráfico de barras com cores personalizadas
ax4 = plt.subplot(gs[4], facecolor='#2E2E2E')
ax4.bar(segments, average_saturations, color=bar_colors)
ax4.set_title('Saturação', color='white')
ax4.tick_params(axis='x', colors='white')
ax4.set_yticks([])
ax4.set_yticklabels([])


fig.set_size_inches(width, height, forward=True)
# Salvar a primeira página no PDF
pp.savefig(orientation='portrait')

# Fechar a figura atual para começar uma nova
plt.close()


# Create a grayscale scale of the image to segment into quartiles of brightness
gray_image = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2GRAY)
_, bins = np.histogram(gray_image.flatten(), bins=4)

# Name of the palettes
palette_names = ['Brancos', 'Luzes', 'Meio-Tons', 'Areas Escuras']

# Neutral gray background color (in RGB)
neutral_gray = np.array([46, 46, 46], dtype=np.uint8)

segment_images = []
segment_palettes = []

# Crie as paletas para cada segmento
for i in range(3, -1, -1):
    mask = cv2.inRange(gray_image, bins[i], bins[i + 1])
    blurred_mask = cv2.GaussianBlur(mask, (9, 9), 0)  # Aplicamos o filtro GaussianBlur para suavizar a máscara
    segmented_image = cv2.bitwise_and(image, image, mask=blurred_mask)
    background_image = np.ones_like(image, dtype=np.uint8) * neutral_gray
    masked_image = cv2.bitwise_and(background_image, background_image, mask=cv2.bitwise_not(blurred_mask))
    final_image = cv2.add(masked_image, segmented_image)

    palette = cluster_image(segmented_image, n_clusters=8)

    # # Definir a cor de fundo
    # fig.patch.set_facecolor('#2E2E2E')
    segment_images.append(final_image)
    segment_palettes.append(palette)

# Adicione novos plots para a segunda página
plt.figure(figsize=(width, height))

# Criar a figura e os subplots
fig = plt.figure(figsize=(width, height), facecolor='#2E2E2E')
gs = gridspec.GridSpec(4, 1, height_ratios=[4, 1, 4, 1])


plot_image(segment_images[0], gs[0], palette_names[0])
plot_image(segment_palettes[0], gs[1], '')
plot_image(segment_images[1], gs[2], palette_names[1])
plot_image(segment_palettes[1], gs[3], '')

# Espaço extra para evitar a superposição
# plt.tight_layout()
plt.subplots_adjust(hspace=1)

fig.set_size_inches(width, height, forward=True)
# Salvar a segunda página no PDF
pp.savefig(orientation='portrait')
plt.close()

# Adicione novos plots para a terceira página
plt.figure(figsize=(width, height))

# Criar a figura e os subplots
fig = plt.figure(figsize=(width, height), facecolor='#2E2E2E')
gs = gridspec.GridSpec(4, 1, height_ratios=[4, 1, 4, 1])


plot_image(segment_images[2], gs[0], palette_names[2])
plot_image(segment_palettes[2], gs[1], '')
plot_image(segment_images[3], gs[2], palette_names[3])
plot_image(segment_palettes[3], gs[3], '')

# Espaço extra para evitar a superposição
# plt.tight_layout()
plt.subplots_adjust(hspace=1)

fig.set_size_inches(width, height, forward=True)
# Salvar a segunda página no PDF
pp.savefig(orientation='portrait')
plt.close()


pp.close()