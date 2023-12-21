from selenium import webdriver
from selenium.webdriver.common.by import By
# from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import time


chrome_options = Options()
chrome_options.add_experimental_option("detach", True)


driver = webdriver.Chrome(options=chrome_options)

# Navigate to the AI2-THOR demo website
# driver.get('https://ai2thor.allenai.org/demo/')


driver.get('file:///home/ahmedjaafar/web/scene1.html')
driver.implicitly_wait(1)

# Wait for the iframe to load
wait = WebDriverWait(driver, 10)
wait.until(EC.frame_to_be_available_and_switch_to_it((By.ID, 'myIframe')))

time.sleep(12.25)

ithor_box = driver.find_element(By.CLASS_NAME, "ant-checkbox-input")
ithor_box.click()

# time.sleep(1)
map = driver.find_elements(By.CLASS_NAME, "ant-card-cover")
driver.execute_script("arguments[0].click();", map[5])


breakpoint()


# Close the browser
driver.quit()