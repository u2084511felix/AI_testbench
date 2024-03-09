"""
in python select all of these 

"""

from playwright.sync_api import sync_playwright

def main():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        context = browser.new_context()
        page = context.new_page()
        page.goto('https://www.google.com/')
        page.fill('input[name="q"]', 'playwright python')
        page.press('input[name="q"]', 'Enter')
        page.wait_for_load_state('load')
        page.screenshot(path='example.png')
        browser.close()