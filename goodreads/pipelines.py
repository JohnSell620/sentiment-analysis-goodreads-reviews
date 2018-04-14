from scrapy.exceptions import DropItem
from twisted.enterprise import adbapi
from datetime import datetime
from hashlib import md5
import logging


class RequiredFieldsPipeline(object):
    required_fields = [
        'title','link_url','book_url','reviewDate','user','review','genre'
    ]

    def process_item(self, item, spider):
        for field in self.required_fields:
            if not item.get(field):
                raise DropItem("Field '%s' missing: %r" % (field, item))
            return item


class RequiredWordsPipeline(object):
    """Only store reviews with certain keywords"""

    # put all words in lowercase
    words_to_filter = ['']  # e.g., 'positive','Dostoyevsky', etc.

    def process_item(self, item, spider):
        count = 0
        for word in self.words_to_filter:
            rw = item.get('review') or ''
            if word not in rw.lower():
				count += 1
        if count == len(self.words_to_filter):
			raise DropItem("Review does not contain keyword(s).")
    	else:
			return item


class MySQLStorePipeline(object):
    """Uses Twisted's asynchronous database API."""

    def __init__(self, dbpool):
        self.dbpool = dbpool

    @classmethod
    def from_settings(cls, settings):
        dbargs = dict(
            host=settings['MYSQL_HOST'],
			port=settings['MYSQL_PORT'],
            db=settings['MYSQL_DBNAME'],
            user=settings['MYSQL_USER'],
            passwd=settings['MYSQL_PASSWD'],
            charset='utf8',
            use_unicode=True,
        )
        dbpool = adbapi.ConnectionPool('MySQLdb', **dbargs)
        return cls(dbpool)

    def process_item(self, item, spider):
        d = self.dbpool.runInteraction(self._do_upsert, item, spider)
        d.addErrback(self._handle_error, item, spider)
        d.addBoth(lambda _: item)
        return d

    def _do_upsert(self, conn, item, spider):
        """Perform an insert or update."""
        guid = self._get_guid(item)
        now = datetime.utcnow().replace(microsecond=0).isoformat(' ')

        conn.execute("""SELECT EXISTS(
            SELECT 1 FROM reviews WHERE guid = %s
        )""", (guid, ))
        ret = conn.fetchone()[0]

        if ret:
            conn.execute("""
                UPDATE reviews
                SET title=%s genre=%s link_url=%s book_url=%s, user=%s, reviewDate=%s, review=%s
                WHERE guid=%s
            """, (item['title'], item['genre'], item['link_url'], item['book_url'], item['user'], item['reviewDate'], item['review'], now, guid))
            spider.log("Item updated in db: %s %r" % (guid, item))
        else:
            conn.execute("""
                INSERT INTO reviews (guid, title, genre, link_url, book_url, user, reviewDate, review)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (guid, item['title'], item['genre'], item['link_url'], item['book_url'], item['user'], item['reviewDate'], item['review'], now))
            spider.log("Item stored in db: %s %r" % (guid, item))

    def _handle_error(self, failure, item, spider):
		"""Handle occurred on db interaction."""
		logger = logging.getLogger()
		logger.warning("Warning: query failed.")

    def _get_guid(self, item):
		"""Generates an unique identifier for a given item."""
		return md5(item['url']).hexdigest()
