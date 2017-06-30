#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: YongfengLi
"""

import scrapy
from scrapy.selector import Selector
from scrapy.crawler import CrawlerRunner  
from twisted.internet import reactor  
from scrapy.settings import Settings  
from scrapy.http import Request 
import sys
from chessspider.items import ChessspiderItem 

rg = list(range(1,1001))
#urlrg = ["http://www.stqiyuan.com/m_game_list.asp?action=&page=" + str(i) for i in rg]
urlrg = ["http://www.01xq.com/e_game_list.asp?page=" + str(i) for i in rg]

class Chessdataspider(scrapy.Spider):
    name = "chessdataspider"
    download_delay = 2
    allowed_domains = ["www.01xq.com"]
    start_urls = urlrg
    

    def parse(self, response):
        hxs = Selector(response)
        iframes = hxs.xpath("//div[@id]").extract()
        is_item = False
        for iframe in iframes:
            item = self.parse_item(iframe)
            if len(item["move"]) > 0:
                is_item = True
                item["url"] = response.url
                yield item
        
        if not is_item and response.url.find("e_game_view.asp") < 0:
            for url in hxs.xpath("//a/@href").extract():
                if url.find("e_game_view.asp") >= 0:
                    print("ok url:",url)
                    if url.find("http://www.01xq.com") >= 0:
                        yield Request(url, callback=self.parse)
                    else:
                        yield Request("http://www.01xq.com/" + url, callback=self.parse)
                else:
                    print("error url", url)

    def get_Data(self,data,pattern):
        result = []  
        start_index = 0  
        for _ in range(10):  
            start = data.find("[{}]".format(pattern), start_index, len(data))
            if start < 0:  
                break  
            start += len("[{}]".format(pattern))  
            end = data.find("[/{}]".format(pattern), start_index, len(data))  
            if start < end:  
                result.append(data[start:end])  
            start_index = end + len("[/{}]".format(pattern))  
        if len(result) == 0:  
            return ""  
        return max(result, key=lambda k: len(result))  


    def parse_item(self,data):
        item = ChessspiderItem()
        item["move"] = self.get_Data(data,"DHJHtmlXQ_34")
        item["conclusion"] = self.get_Data(data, "DHJHtmlXQ_28")
        item["redplayer"] = self.get_Data(data, "DHJHtmlXQ_18")
        item["blackplayer"] = self.get_Data(data,"DHJHtmlXQ_22")

        print("0"+item["move"])
        return item
