# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# http://doc.scrapy.org/en/latest/topics/items.html
# defined by YongfengLi
#

import scrapy


class ChessspiderItem(scrapy.Item):
    # define the fields for your item here like:
    url = scrapy.Field()
    move = scrapy.Field()
    redplayer = scrapy.Field()
    blackplayer = scrapy.Field()
    conclusion = scrapy.Field()
    pass

