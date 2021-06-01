# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy

class HarassmentItem(scrapy.Item):
    title = scrapy.Field()
    date = scrapy.Field()
    reason = scrapy.Field()
    content = scrapy.Field()
