import requests

token = "secret_nCvUT5LSQM9Fq3tlnfeLGaipD4xCrXoBZueaog3FGD8"
# token = "test"
headers = {
    "Notion-Version": "2021-08-16",
    'Authorization': 'Bearer ' + token,
}

database_id = "08d60527f5a84642a3a185954af89287"
database_id = "cdd07dfe408e4538bbb3ce649079d85f"
database_id = "3cf1215ff6244ec8bc4ca0b69277f315"
database_id = "4fe7cc1931c949eaae9feeccf3b548fe"
database_id = "5c1ba472f4e8415295fcaf03264a9ef5"
https://www.notion.so/zhangyiteng/test-cdd07dfe408e4538bbb3ce649079d85f
https://www.notion.so/zhangyiteng/4fe7cc1931c949eaae9feeccf3b548fe

def add_hyphen(s):
    assert len(s) == 32
    return "-".join([s[:8], s[8:12], s[12:16], s[16:20], s[20:]])

url_notion = 'https://api.notion.com/v1/databases/' + add_hyphen(database_id) + '/query'
url_notion = 'https://api.notion.com/v1/databases/' + database_id + '/query'

# https://www.notion.so/zhangyiteng/test-cdd07dfe408e4538bbb3ce649079d85f

res_notion = requests.post(url_notion, headers=headers)

https://www.notion.so/zhangyiteng/test-cdd07dfe408e4538bbb3ce649079d85f
https://www.notion.so/zhangyiteng/39c1a2eb8717452094696e5b0891ba10?v=3cf1215ff6244ec8bc4ca0b69277f315

res_notion.json()

page_id = b38f5e00f46c412d9108a69a27394489
url_notion = "https://api.notion.com/v1/pages/" + page_id

body = {
"properties": {
    ""
    }
        }
