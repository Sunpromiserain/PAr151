import numpy as np
import cv2
from scipy.fftpack import dct, idct
import matplotlib.pyplot as plt

# Opérations DCT et IDCT
def block_dct(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def block_idct(block):
    return idct(idct(block.T, norm='ortho').T, norm='ortho')

# Calcul de la médiane des coefficients basse fréquence
def compute_median_low_freq(block, zigzag_indices):
    low_freq = [block[x, y] for x, y in zigzag_indices]
    return np.median(low_freq)



def embed_watermark(image, watermark, threshold=80, k=12, z=2):
    block_size = 8
    h, w = image.shape
    # image -= 128   #  il ne faut pas -128

    # 分块
    blocks = divide_blocks(image, block_size)
    # 打印一个想要的block # 3行6列
    # print(blocks[134])

    # 计算所有块的 DCT 并存储
    dct_matrix = [
        block_dct(block) for block in blocks
    ]
    #打印dct——block
    # print(dct_matrix[134])

    watermark_flat = watermark.flatten()
    
    key = np.random.randint(0, 100000)  # Générer une clé aléatoire entre 0 et 100000
    rng = np.random.default_rng(seed=key)

    # Appliquer une permutation aléatoire sur le watermark aplati
    # watermark_flat = rng.permutation(watermark_flat)
    wm_idx = 0

    # 定义 Zig-Zag 序列的索引
    zigzag_indices = [
        (0, 1), (1, 0), (2, 0), (1, 1), (0, 2), (0, 3), (1, 2), (2, 1), (3, 0)
    ]

    # 计算修改参数
    modification_params = []
    for dct_block in dct_matrix:
        dc_coefficient = dct_block[0, 0]
        med = compute_median_low_freq(dct_block, zigzag_indices)
        abs_dc = abs(dc_coefficient)

        if abs(med) > abs_dc:
            med = dc_coefficient

        if abs_dc > 1000 or abs_dc < 1:
            modification_param = min(abs(z * med),20)
            modification_param = max(modification_param,0.5)
        else:
            modification_param = min(abs(z * (dc_coefficient - med) / dc_coefficient),20)
            modification_param = max(modification_param,0.5)

        modification_params.append(modification_param)

    # 插入水印
    for b_idx, dct_block in enumerate(dct_matrix):
        if wm_idx >= len(watermark_flat):
            break

        P = b_idx // (w // block_size)
        Q = b_idx % (w // block_size)

        wm_bit = watermark_flat[wm_idx]
        targets = {'LR': (3, 7),
                  'UD': (4, 6),
                  'DU': (6, 4),
                  'RL': (7, 3)
                  } # Exemple coefficients 

        # Determinier la position du block adjacent
        neighbors = {
            'LR': (P, Q + 1),
            'UD': (P + 1, Q),
            'DU': (P - 1, Q),
            'RL': (P, Q - 1)
        }

        if P % 2 == 0 and Q % 2 == 0:
            neighbor = neighbors['LR']
            target = targets['LR']
        elif P % 2 == 1 and Q % 2 == 0:
            neighbor = neighbors['DU']
            target = targets['DU']
        elif P % 2 == 0 and Q % 2 == 1:
            neighbor = neighbors['UD']
            target = targets['UD']
        else:
            neighbor = neighbors['RL']
            target = targets['RL']

        # 检查相邻块是否在图像范围内
        if 0 <= neighbor[0] < h // block_size and 0 <= neighbor[1] < w // block_size:
            neighbor_idx = neighbor[0] * (w // block_size) + neighbor[1]
            neighbor_block = dct_matrix[neighbor_idx]

            # 计算差异
            diff = dct_block[target] - neighbor_block[target]
            param = modification_params[b_idx]
            # if P == p and Q == q :                                                    # test pour une position
            #     print(f"[{P},{Q}]Before: diff={diff}, param={param}")
            #     print(f"dct_block_target: {dct_block[target]}")
            # print(f"Before: diff={diff}, param={param}")

            

            # Intégration du tatouage
            if param != 0 :
                if wm_bit == 1:
                    if diff > threshold - k:
                        while diff > threshold - k:
                            dct_block[target] -= param
                            diff -= param
                    elif k > diff > -threshold / 2:
                        while diff < k:
                            dct_block[target] += param
                            diff += param
                    elif diff < -threshold / 2:
                        while diff > -threshold - k:
                            dct_block[target] -= param
                            diff -= param
                else:  # wm_bit == 0
                    if diff > threshold / 2:
                        while diff < threshold + k:
                            dct_block[target] += param
                            diff += param
                    elif -k < diff < threshold / 2:
                        while diff > -k:
                            dct_block[target] -= param
                            diff -= param
                    elif diff < k - threshold:
                        while diff < -threshold + k:
                            dct_block[target] += param
                            diff += param
            
        wm_idx += 1
 
    # iDCT
    for b_idx, dct_block in enumerate(dct_matrix):
        blocks[b_idx] = block_idct(dct_block)

    # Réassembler les blocks
    watermarked_image = merge_blocks(blocks, h, w, block_size)
    # watermarked_image += 128
    return np.clip(watermarked_image, 0, 255).astype(np.uint8)   # normalisation en 0-255，（-20→0，300→255）


# Fonction de division en blocs
def divide_blocks(image, block_size):
    h, w = image.shape
    blocks = [
        image[i:i + block_size, j:j + block_size]
        for i in range(0, h, block_size)
        for j in range(0, w, block_size)
    ]
    return blocks

# Fonction de fusion des blocs
def merge_blocks(blocks, height, width, block_size):
    image = np.zeros((height, width))
    idx = 0
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            image[i:i + block_size, j:j + block_size] = blocks[idx]
            idx += 1
    return image

def extract_watermark(watermarked_image, block_size=8, threshold=80, k = 12, key=None):
    # Step 1: moins 128
    watermarked_image = watermarked_image.astype(np.float32) - 128   # cette ligne n'a pas d'effet sur le resultat
    
    h, w = watermarked_image.shape
    R = h // block_size
    C = w // block_size

    # initialisation
    extracted_watermark = np.zeros((R, C), dtype=np.uint8)
    diff_matrice = np.zeros((R, C), dtype=np.uint8)
    # division
    blocks = divide_blocks(watermarked_image, block_size)

    # calculation et stokage
    dct_matrix = [
        block_dct(block) for block in blocks
    ]

    # insertion du tatouage
    for b_idx, dct_block in enumerate(dct_matrix):

        P = b_idx // (w // block_size)
        Q = b_idx % (w // block_size)

        targets = {'LR': (3, 7),
                  'UD': (4, 6),
                  'DU': (6, 4),
                  'RL': (7, 3)
                  } # exemple des targets

        # determiner la position interbloc
        neighbors = {
            'LR': (P, Q + 1),
            'UD': (P + 1, Q),
            'DU': (P - 1, Q),
            'RL': (P, Q - 1)
        }

        # choix de la direction
        if P % 2 == 0 and Q % 2 == 0:
            neighbor = neighbors['LR']
            target = targets['LR']
        elif P % 2 == 1 and Q % 2 == 0:
            neighbor = neighbors['DU']
            target = targets['DU']
        elif P % 2 == 0 and Q % 2 == 1:
            neighbor = neighbors['UD']
            target = targets['UD']
        else:
            neighbor = neighbors['RL']
            target = targets['RL']

        # verifier il n'a pas de hors limites
        
        if 0 <= neighbor[0] < h // block_size and 0 <= neighbor[1] < w // block_size:
            neighbor_idx = neighbor[0] * (w // block_size) + neighbor[1]
            neighbor_block = dct_matrix[neighbor_idx]

            # calculer diff
            diff = dct_block[target] - neighbor_block[target]
            diff_matrice[P,Q] = diff
            if diff > threshold + k or (-threshold + k < diff < 0) :
                extracted_watermark[P,Q] = 0
            elif diff < -threshold - k or 0 < diff < threshold - k :
                extracted_watermark[P,Q] = 1
            else :
                extracted_watermark[P,Q] = 0.5
            # if P == p and Q == q:
            #     print(f"diff_[{P},{Q}]:{diff}")
            #     print(f"dct_block_target: {dct_block[target]},neighbor_block_target: {neighbor_block[target]}")
    
    # Step 5: key pseudo random
    if key is not None:
        # pour obtenir le tatouage original

        rng = np.random.default_rng(seed=key)
        indices = rng.permutation(R * C)
        inverse_indices = np.argsort(indices)
        extracted_watermark = extracted_watermark.flatten()[inverse_indices].reshape(R, C)


    # design diff_matrice(optionnel)
    #np.set_printoptions(threshold=np.inf, linewidth=200)
    #print(diff_matrice)
    plt.figure(figsize=(8, 8))
    plt.title("Diff Matrix Visualization")
    plt.imshow(diff_matrice, cmap='gray', interpolation='nearest')
    plt.colorbar()  # barre de couleur
    plt.axis('off')  # axe off
    # plt.show()

    return extracted_watermark


# Programme principal
if __name__ == "__main__":
    # lire l'image
    cover_image = cv2.imread("lena_image.jpg", cv2.IMREAD_GRAYSCALE)  # 512*512

    block_size = 8
    h, w = cover_image.shape

    # Lire l'image en niveaux de gris
    watermark_image = cv2.imread('watermarking6.jpg', cv2.IMREAD_GRAYSCALE)  #64*64

    # Définir un seuil pour la binarisation
    threshold1 = 80  # Les pixels <100 deviennent 0, et >=100 deviennent 1

    (p,q) = (15,50)    # ici on peut modifier le point observable (au debut pour trouver ou est le probleme)
    watermark = (watermark_image >= threshold1).astype(np.uint8)
    # print(f"watermark: {watermark[p,q]}")

    # integrer le tatouage
    watermarked_image = embed_watermark(cover_image, watermark)
    # un exemple pour ajouter un bloc noire 
    # watermarked_image[128:256, 128:256] = 0
    # 假设 watermarked_image 是 numpy.ndarray 类型
    cv2.imwrite("output/watermarked_image.png", watermarked_image)


    # extraction du tatouage
    extracted_watermark = extract_watermark(watermarked_image,key=None)
    # print(f"extracted_watermark:{extracted_watermark[p,q]}")




    # trouver le nombre des ecarts des pixels
    different_positions = np.argwhere(extracted_watermark != watermark)

    # print
    print("l'ecart des pixels:",len(different_positions))
    '''
    l'ecart des pixels: nombre : 28  [[14, 31], [15, 47], [18, 39], [20, 13], [21, 50], [25, 33], [26, 44], [33, 14], [33, 43], [34, 20], [34, 24],
    [36, 25], [38, 22], [39, 63], [43, 9], [44, 13], [44, 15], [46, 16], [51, 42], [54, 18], [54, 19], [54, 20], [55, 5], [58, 18], 
    [59, 17], [60, 16], [61, 17], [63, 16]] ,different_positions.tolist()
    '''
    # imshow
    plt.figure(figsize=(15, 7))

    # image originale
    plt.subplot(2, 2, 1)
    plt.title('Original Image')
    plt.imshow(cover_image, cmap='gray')
    plt.axis('off')  # 关闭坐标轴
  

    # image tatougee
    plt.subplot(2, 2, 2)
    plt.title('Watermarked Image')
    plt.imshow(watermarked_image, cmap='gray')
    plt.axis('off')  # 关闭坐标轴

    # image du tatouage
    plt.subplot(2, 2, 3)
    plt.title("Watermark")
    plt.imshow(watermark, cmap='gray')  # 确保显示范围在[0,1]
    plt.axis('off')  # 关闭坐标轴

    # tatouage extract
    plt.subplot(2, 2, 4)
    plt.title("Extracted Watermark")
    plt.imshow(extracted_watermark, cmap='gray', vmin=0, vmax=1)  # 确保显示范围在[0,1]
    plt.axis('off')  # 关闭坐标轴

    # image show
    plt.show()

    # 
    np.set_printoptions(threshold=np.inf, linewidth=200)

    print()

    '''
    # pour obtenir le tatouage original
    inverse_indices = np.argsort(permuted_indices)
    recovered_array = permuted_watermark[inverse_indices]
    '''