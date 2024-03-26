import asyncio
import re
import random

import aiohttp
import pandas as pd


async def get_many_results(session: aiohttp.ClientSession, query: str, random_delay: int | None = None) -> list:
    """
    Get many results from a SPARQL query.
    """
    url = "https://query.wikidata.org/sparql"
    params = {"format": "json", "query": query}
    if random_delay is not None:
        await asyncio.sleep(random.random() * random_delay)
    async with session.get(url, params=params) as response:
        try:
            data = await response.json()
        except aiohttp.ContentTypeError as ex:
            text = await response.text()
            raise RuntimeError(f"Error while getting results: {text}") from ex
        return data["results"]["bindings"]


async def get_one_result(session: aiohttp.ClientSession, query: str) -> str | None:
    """
    Get one result from a SPARQL query.
    """
    results = await get_many_results(session, query)
    if len(results) == 0:
        return None
    return results[0]


async def get_item_label(session: aiohttp.ClientSession, item: str) -> str:
    query = """
    SELECT ?itemLabel WHERE {{
      wd:{item} rdfs:label ?itemLabel.
      FILTER(LANG(?itemLabel) = "en")
    }}
    """.format(item=item)
    result = await get_one_result(session, query)
    return result["itemLabel"]["value"]


def sanitize_description(description: str) -> str:
    """
    Sanitize a description.
    """
    regex = re.compile(r"\s*\([^)]*\)")
    description = regex.sub("", description)
    return description


async def get_humans(session: aiohttp.ClientSession) -> None:
    from_year = 1900
    to_year = 2004
    n_years_per_chunk = 5
    all_humans = []
    all_queries = []
    for i in range(from_year, to_year, n_years_per_chunk):
        query = """
        SELECT ?item ?itemLabel ?itemDesc ?siteLinks WHERE {{
            ?item wdt:P31 wd:Q5.  # We want a human
            FILTER EXISTS {{
                ?item wdt:P18 ?itemPicture.  # that has a picture
            }}
            ?item wikibase:sitelinks ?siteLinks.
            FILTER (?siteLinks >= 30).  # We use the number of site links as a proxy for notability
            ?item rdfs:label ?itemLabel.  # We want the label in English; we don't use the labelling service because it's too slow
            FILTER(LANG(?itemLabel) = "en").
            ?item wdt:P569 ?dateOfBirth. hint:Prior hint:rangeSafe true.
            FILTER ("{from_year}-00-00"^^xsd:dateTime <= ?dateOfBirth && ?dateOfBirth <= "{to_year}-00-00"^^xsd:dateTime).
            OPTIONAL {{
                ?item schema:description ?itemDesc.
                FILTER(LANG(?itemDesc) = "en").
            }}
            ?item wdt:P27 ?citizenship.
            FILTER(?citizenship NOT IN (wd:Q7318, wd:Q423)).  # We don't want people from Nazi Germany or North Korea
        }}
        """.format(from_year=i, to_year=i + n_years_per_chunk)
        all_queries.append(query)
    query_results = await asyncio.gather(*[get_many_results(session, query, random_delay=60) for query in all_queries])
    for query_result in query_results:
        for elem in query_result:
            item = elem["item"]["value"].split("/")[-1]
            item_label = elem["itemLabel"]["value"]
            item_desc = elem.get("itemDesc", {}).get("value", None)
            if item_desc is not None:
                item_desc = sanitize_description(item_desc)
            site_links = int(elem["siteLinks"]["value"])
            all_humans.append((item, item_label, item_desc, site_links))
    df = pd.DataFrame(all_humans, columns=["item", "itemLabel", "itemDesc", "siteLinks"])
    df = df.drop_duplicates(subset=["item"], keep="first")
    df = df.drop_duplicates(subset=["itemLabel"], keep=False)
    df = df.sort_values(by=["siteLinks"], ascending=False)
    df.to_csv(f"all_humans.csv", index=False)
    prompts = []
    for _, row in df.iterrows():
        item_label, item_desc = row["itemLabel"], row["itemDesc"]
        if item_desc is None:
            item_desc = ""
        prompts.append(f"{item_label}|{item_desc}")
    with open(f"all_humans.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(prompts))


async def main():
    async with aiohttp.ClientSession() as session:
        await get_humans(session)


if __name__ == "__main__":
    asyncio.run(main())