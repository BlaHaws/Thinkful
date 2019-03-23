import scrapy

class CFSSpider(scrapy.Spider):
	name = 'sundfun'
	start_urls = ['https://www.sundbergolpinmortuary.com/listings']
	allowed_domains = ['sundbergolpinmortuary.com']
	
	def parse(self, response):
		for url in response.css('span.obitlist-title > a::attr(href)').extract():
			yield scrapy.Request(response.urljoin(url), self.parse_listing)
			
	def parse_listing(self, response):
		item = {
			'Name': response.css('h1.obitnameV3::text').extract_first() or '',
			'Date of Birth': response.css('span.dob::text').extract_first() or '',
			'Date of Death': response.css('span.dod::text').extract_first() or '',
			'Obituary': ' '.join(response.css('div#obtext ::text').extract()).strip().replace('\n', '')
		}
		
		yield item