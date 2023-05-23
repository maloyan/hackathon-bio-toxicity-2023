import requests

url_dataset_category = "https://toxric.bioinforai.tech/jk/DataCollections/getToxicityCategoryInfo?pid=0"
response = requests.get(url_dataset_category)
if response.status_code == 200:
    data = response.json()
    ids_and_categories_dataset_category = [(item['id'], item['category']) for item in data['data']['list']]
    print(ids_and_categories_dataset_category)

url_dataset = "https://toxric.bioinforai.tech/jk/DownloadController/getCategoryDetailedInfo?type=1&pageSize=1000&pageNo=1&pid="
ids_and_categories = []
for id, category in ids_and_categories_dataset_category:
    response = requests.get(f"{url_dataset}{id}")
    if response.status_code == 200:
        data = response.json()
        ids_and_categories.extend([(category, item['category'], item['id']) for item in data['data']['list']])

for tox_group, experiment, toxicity_id in ids_and_categories:
    url = f"https://toxric.bioinforai.tech/jk/DownloadController/DownloadToxicityInfo?toxicityId={toxicity_id}"
    response = requests.get(url)
    if response.status_code == 200:
        # Save the content to a file
        with open(f"{tox_group}_{experiment}.csv", "wb") as file:
            file.write(response.content)
        print(f"File saved as {tox_group}_{experiment}.csv")

