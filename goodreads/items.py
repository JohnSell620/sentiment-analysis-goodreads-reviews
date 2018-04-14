from scrapy.item import Item, Field

class GoodreadsItem(Item):
    link_url = Field()
    reviewDate = Field()
    user = Field()
    review = Field()
    title = Field()
    book_url = Field()
    genre = Field()
