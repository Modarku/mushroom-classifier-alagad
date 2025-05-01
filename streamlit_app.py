import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model('models/mushroom_classification_mobilenetv3small.keras')

species = ["Agaricus", "Amanita", "Boletus", "Cortinarius", "Entoloma", "Hygrocybe", "Lactarius", "Russula", "Suillus"]
species_details = {
    "Agaricus": '''**Edibility:** Includes popular edible species like Button Mushroom.  
        **Appearance:** Smooth caps, pink to dark brown gills.  
        **Habitat:** Found in meadows, grasslands, and forests.''',
    "Amanita": '''**Edibility:** :red[Caution:] Some deadly, like Death Cap, and some edible.  
        **Appearance:** Smooth, often brightly colored caps with rings and volvas.  
        **Habitat:** Common in forests, often mycorrhizal with trees.''',
    "Boletus": '''**Edibility:** Many edible, such as Boletus edulis (porcini).  
        **Appearance:** Thick, meaty caps, spongy underside.  
        **Habitat:** Ectomycorrhizal, often under pines and oaks.''',
    "Cortinarius": '''**Edibility:** :red[Caution:] Some edible, but others, like the rare deadly webcap is highly toxic.  
        **Appearance:** Rusty-brown caps, web-like veil on stem.  
        **Habitat:** Found in forests, particularly under conifers.''',
    "Entoloma": '''**Edibility:** :red[Caution:] Most are toxic or mildly poisonous.  
        **Appearance:** Conical to bell-shaped caps, often pink gills.  
        **Habitat:** Common in woodlands and grassy areas.''',
    "Hygrocybe": '''**Edibility:** :red[Caution:] Generally not considered edible, often toxic.  
        **Appearance:** Bright, waxy caps with vibrant colors.  
        **Habitat:** Found in grasslands and woodland edges.''',
    "Lactarius": '''**Edibility:** :red[Caution:] Many edible species, but some having peppery or even burning tastes are toxic.  
        **Appearance:** Milky latex from the cap when cut.  
        **Habitat:** Often in forests, associated with certain trees.''',
    "Russula": '''**Edibility:** :red[Caution:] some species are safe, but hot and peppery taste may be poisonous.  
        **Appearance:** Brightly colored caps, brittle flesh.  
        **Habitat:** Found in woodlands, forming symbiotic relationships with trees.''',
    "Suillus": '''**Edibility:** Mostly edible, prized in certain regions.  
        **Appearance:** Slimy caps, with glandular dots on stems.  
        **Habitat:** Mycorrhizal, often with pines and other conifers.''',
}

st.title('Mushroom Species Classifier')
st.write("Upload an image of a mushroom to classify its species.")

uploaded_image = st.file_uploader("Choose an image...", type="jpg")

if uploaded_image is not None:
    img = Image.open(uploaded_image)
    st.image(img, caption='Uploaded Mushroom Image', use_container_width=True)

    IMG_SIZE = 224
    img = img.resize((IMG_SIZE, IMG_SIZE)) 
    img_array = tf.keras.preprocessing.image.img_to_array(img) 
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)

    predictions = model.predict(img_array)

    print(predictions)

    predicted_class = np.argmax(predictions)
    predicted_species = species[predicted_class]
    
    st.write(f"The predicted species is: **{predicted_species}**")
    st.markdown(species_details.get(predicted_species, "Details not available"))

    np.set_printoptions(precision=4, suppress=True)
    percentages = predictions[0] * 100
    for species, pct in zip(species, percentages):
        st.write(f"{species}: {pct:.2f}%")
