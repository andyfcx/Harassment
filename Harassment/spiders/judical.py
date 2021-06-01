import scrapy
from Harassment.items import HarassmentItem
from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor
from Harassment.items import HarassmentItem
from fake_useragent import UserAgent

ua = UserAgent(verify_ssl=False)

def validate(self, source):
    # these fields are the minimum required as cannot be hardcoded
    data = {"__VIEWSTATEGENERATOR": source.xpath("//*[@id='__VIEWSTATEGENERATOR']/@value")[0].extract(),
            "__EVENTVALIDATION": source.xpath("//*[@id='__EVENTVALIDATION']/@value")[0].extract(),
            "__VIEWSTATE": source.xpath("//*[@id='__VIEWSTATE']/@value")[0].extract(),
            " __VIEWSTATEENCRYPTED": source.xpath("//*[@id='__VIEWSTATEENCRYPTED']/@value")[0].extract()}
    return data


class TrackSpider(CrawlSpider):
    name = 'track'
    allowed_domains = ['law.judicial.gov.tw']
    start_urls = [
        'https://law.judicial.gov.tw/FJUD/qryresultlst.aspx?ty=JUDBOOK&q=3c91639cdda346a39200d4cf2436ea74',  # 跟蹤
        'https://law.judicial.gov.tw/FJUD/qryresultlst.aspx?ty=JUDBOOK&q=78a056158e5f327c5d88ef99bcbed134'  # 騷擾
    ]

    # https://law.judicial.gov.tw/FJUD/qryresultlst.aspx?q=78a056158e5f327c5d88ef99bcbed134&sort=DS&page=2&ot=in

    rules = [
        Rule(LinkExtractor(allow='https://law.judicial.gov.tw/FJUD/qryresultlst.aspx?q=.*&sort=DS&page=.*'),
             callback='parse', follow=True),
    ]

    def parse(self, response):

        judgement_links = response.xpath("//a/@href")

        for judgement_link in judgement_links:
            url = judgement_link.extract()

            yield scrapy.Request("https://law.judicial.gov.tw"+url, callback=self.parse_item)

    def parse_item(self, response):
        item = HarassmentItem()
        item.title = response.xpath('//*[@id="jud"]/div[1]/div[2]/text()')
        item.date = response.xpath('//*[@id="jud"]/div[2]/div[2]/text()')
        item.reason = response.xpath('//*[@id="jud"]/div[3]/div[2]/text()')
        item.content = response.xpath(
            '//*[@id="jud"]/div[4]/div/table/tbody/tr/td[1]/div[2]/text()')
