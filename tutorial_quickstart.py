import fiftyone as fo
import fiftyone.brain as fob
import fiftyone.zoo as foz
import numpy as np

# Step 1: Load your data into FiftyOne
dataset = foz.load_zoo_dataset("quickstart")

# Steps 2 and 3: Compute embeddings and create a similarity index
elasticsearch_index = fob.compute_similarity(
    dataset,
    model="clip-vit-base32-torch",
    brain_key="elasticsearch_index",
    backend="elasticsearch",
    hosts="http://localhost:9200",
    username="elastic",
    password="Y5OdqZyHYXFo2UweLReg",
    #embeddings=False,
    metric="cosine",
)

session = fo.launch_app(dataset)

# Query by vector
# query = np.random.rand(512)  # matches the dimension of CLIP embeddings
# view1 = dataset.sort_by_similarity(query, k=10, brain_key="elasticsearch_index")

    # # Query by sample ID
    # query = dataset.first().id
    # view = dataset.sort_by_similarity(query, k=10, brain_key="elasticsearch_index")`

# # Query by a list of IDs
# query = [dataset.first().id, dataset.last().id]
# view = dataset.sort_by_similarity(query, k=10, brain_key="elasticsearch_index")

# Query by text prompt
query = "a photo of a dog"
view2 = dataset.sort_by_similarity(query, k=2, brain_key="elasticsearch_index", hosts="http://localhost:9200")
session.view = view2
#session.view = view2

print(view2)

# should hold until btowser tab is closed
session.wait()

