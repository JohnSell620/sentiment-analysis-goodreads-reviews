import sys
from scrapy import Spider
from scrapy.selector import Selector
from scrapy.http import Request
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import TimeoutException
from goodreads.items import GoodreadsItem


class GoodreadsSpider(Spider):
    name = "goodreads"
    allowed_domains = ['www.goodreads.com']
    start_urls = [
        'https://www.goodreads.com/genres/art',
        # 'https://www.goodreads.com/genres/history',
        # 'https://www.goodreads.com/genres/philosophy',
        # 'https://www.goodreads.com/genres/religion',
        # 'https://www.goodreads.com/genres/science',
    ]


    def __init__(self):
        self.driver = webdriver.Firefox()


    def parse(self,response):
        book_urls = Selector(response).xpath('//div[@class="leftContainer"]/div/div[@class="bigBoxBody"]/div/div/div[@class="leftAlignedImage bookBox"]/div[@class="coverWrapper"]/a/@href')

        for book_url in book_urls:
            # item = GoodreadsItem()
            book_url = book_url.extract()
            url = "https://www.goodreads.com" + book_url
            self.driver.get(url)
            request = Request(url,callback=self.parse2)
            # request.meta['item'] = item
            request.meta['book_url'] = url
            yield request

        self.driver.close()


    def parse2(self,response):
        try:
            timeout = WebDriverWait(self.driver,10)
        except:
            print("Timed out waiting for page load.")
            self.driver.quit()

        title = Selector(response).xpath('//div[@class="leftContainer"]/div/div/div/div/a/img[@id="coverImage"]/@alt')
        genre = Selector(response).xpath('//div[@class="rightContainer"]/div/div/div[@class="bigBoxBody"]/div/div/div[@class="left"]/a/text()')
        reviews = Selector(response).xpath('//div[@id="bookReviews"]/div[@class="friendReviews elementListBrown"]')

        for review in reviews:
            item = GoodreadsItem()
            item['title'] =  title.extract()[0]
            item['book_url'] = response.meta['book_url']
            item['genre'] = genre.extract()
            item['link_url'] = review.xpath(           './/div/div/link/@href').extract()[0]
            item['reviewDate'] = review.xpath(               './/div/div/div/div/a/text()').extract()[0]
            item['user'] = review.xpath(               './/div/div/div/div/span/a/text()').extract()[0]
            item['review'] = review.xpath(               './/div/div/div/div[@class="reviewText stacked"]/span/span[2]/text()').extract()[0]
            yield item
