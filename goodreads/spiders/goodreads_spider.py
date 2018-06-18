from __future__ import print_function

import sys
from scrapy import Spider
from scrapy.selector import Selector
from scrapy.http import Request
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import TimeoutException
from goodreads.items import GoodreadsItem
from HTMLParser import HTMLParser
from w3lib.html import remove_tags
import logging

logger = logging.getLogger(__name__)
logger.setLevel('DEBUG')

class GoodreadsSpider(Spider):
    name = "goodreads"
    allowed_domains = ['www.goodreads.com']
    # crawler will scrape more genres than these listed
    start_urls = [
        'https://www.goodreads.com/genres/art',
        'https://www.goodreads.com/genres/history',
        'https://www.goodreads.com/genres/philosophy',
        'https://www.goodreads.com/genres/religion',
        'https://www.goodreads.com/genres/science',
    ]

    def __init__(self):
        self.driver = webdriver.Firefox()

    def parse(self,response):
        book_urls = Selector(response).xpath('//div[@class="leftContainer"]/\
                    div/div[@class="bigBoxBody"]/div/div/\
                    div[@class="leftAlignedImage bookBox"]/\
                    div[@class="coverWrapper"]/a/@href')

        for book_url in book_urls:
            book_url = book_url.extract()
            url = "https://www.goodreads.com" + book_url
            self.driver.get(url)
            request = Request(url,callback=self.parse2)
            request.meta['book_url'] = url
            yield request

        self.driver.close()

    def parse2(self,response):
        try:
            timeout = WebDriverWait(self.driver,10)
        except:
            print("Timed out waiting for page load.")
            self.driver.quit()

        title = Selector(response).xpath(
                    '//div[@class="leftContainer"]/div/div/div/div/ \
                    a/img[@id="coverImage"]/@alt'
                    )
        genre = Selector(response).xpath(
                    '//div[@class="rightContainer"]/div/div/ \
                    div[@class="bigBoxBody"]/div/div/div[@class="left"]/a/text()'
                    )
        rating = Selector(response).xpath(
                    '//div[@class="leftContainer"]/div/div[@id="metacol"]/ \
                    div[@id="bookMeta"]/span/span[@class="average"]/text()'
                    )
        reviews = Selector(response).xpath(
                    '//div[@id="bookReviews"]/ \
                    div[@class="friendReviews elementListBrown"]'
                    )

        for review in reviews:
            try:
                item = GoodreadsItem()
                item['title'] =  title.extract()[0]
                item['rating'] = rating.extract()[0]
                item['book_url'] = response.meta['book_url']
                item['genre'] = genre.extract()[0]
                item['link_url'] = review.xpath(
                                './/div/div/link/@href').extract()[0]
                item['reviewDate'] = review.xpath(
                                './/div/div/div/div/a/text()').extract()[0]
                item['user'] = review.xpath(
                                './/div/div/div/div/span/a/text()').extract()[0]

                review_text = review.xpath('.//div/div/div/ \
                                div[@class="reviewText stacked"]/span/ \
                                span[2]/text()'
                                ).extract()[0]
                # remove html tags
                item['review'] = remove_tags(review_text)

            except IndexError as e:
                print(e,": title: ",item['title'], "user: ",item['user'])
                logger.error(e.args[0])
                raise

            yield item
