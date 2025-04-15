read me :

test.py -- main fonction pour insertion et extraction du tatouage

exemple.py -- inprimer un exemple du bloc dct 

evaluation.py -- test avec l'attaque WAVES sur l'image tatoué



Dans WAVES\distortions\, on a ajouté un fichier 'attack.py' pour effectuer des attaques sur les images; ce fichier permet de enregistrer 
les images attaquées dans la répertoire 'output'; et ensuite combiner avec 'evaluation.py' peut afficher les résultats de l'image du filigrane extrais
après différentes attaques.


Attention : Un chemin absolu est utilisé dans le code, les utilisateurs ultérieurs doivent donc modifier le nom du chemin lors de son application.
（Tous les chemins sont sous la forme C:\Users\zyr\Desktop\PAr... Nous sommes vraiment désolés pour la gêne occasionnée. Nous le modifierons 
dans les travaux futurs pour en garantir la rigueur.）

Le fichier distortion.py dans le répertoire distortions (qui était à l'origine fourni avec WAVES) a été modifié. C'est parce que la syntaxe 
from ... import... a donné une erreur, donc pour plus de commodité, nous avons adopté la méthode de copie directe des fonctions.

Les fichiers .env et find.py ont également été légèrement modifiés, mais ils n'ont pas été utilisés.


Théoriquement, nous n'avons besoin d'exécuter que les trois fichiers test.py, attack.py et evaluation.py pour obtenir tous les résultats dont nous disposons. 
(attack.py dépend des fonctions pertinentes dans WAVES, vous devrez donc peut-être télécharger des extensions liées à WAVES)

Au fait, notre expérience n'a utilisé que l'image lena_image pour l'opération. Les dossiers principaux et attaqués ont été ajoutés manuellement par nos soins 
pour appliquer l'outil WAVES, mais nous n'avons pas pu exécuter avec succès les fonctions associées (nous avons essayé de contacter l'auteur de WAVES 
mais n'avons reçu aucune réponse). De plus, la bibliothèque d'images utilisée dans l'article se trouve dans le répertoire C:\Users\zyr\Desktop\PAr\main\mscoco.(SUPPRIME)
 Les deux autres, differencedb et dalle3, n'ont pas été copiés car l'un avait une image trop grande (nécessite au moins 500 Go) et l'autre avait un chemin endommagé.

Source dataset utilisé dans l'acticle WAVES :
 • DiffusionDB: the 2m_random_100ksplit of DiffusionDB dataset (Wang et al., 2022), https://huggingface.co/datasets/poloclub/diffusiondb/viewer/2m_random_100k.
 • MS-COCO:thevalidationsplit of the 2017 Microsoft COCOdetection challenge (Lin et al., 2014), http://images.cocodataset.org/zips/val2017.zip.
 • DALL·E3: the train split of the dalle-3-dataset repository on HuggingFace, collected from the LAION share-dalle-3
 discord channel, https://huggingface.co/datasets/laion/dalle-3-dataset

Certains autres fichiers dans WAVES, tels que cli.py, scripts\decode.py, etc., ont été modifiés parce que nous avons essayé d'intégrer notre propre programme de 
filigrane dans l'outil WAVES, mais nous avons échoué.


contact : zhangyurou72@gmail.com ou tanj3475@gmail.com