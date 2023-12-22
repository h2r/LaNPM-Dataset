from selenium import webdriver
from selenium.webdriver.common.by import By
# from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import time
import pandas as pd


# Function to extract text box values and return them as a dictionary
def extract_textbox_values(driver, scene_name):
    data = {'Scene Name': scene_name}
    for i in range(1, 6):  # Assuming there are 5 command text boxes
        textbox = driver.find_element(By.ID, f'command{i}')
        data[f'Command {i}'] = textbox.get_attribute('value')
    return data


# List to store data dictionaries for all scenes
all_scene_data = []


chrome_options = Options()
chrome_options.add_experimental_option("detach", True)


driver = webdriver.Chrome(options=chrome_options)

# driver.get('file:///home/ahmedjaafar/web/scene1.html')
# driver.implicitly_wait(1)

driver.get('file:///home/ahmedjaafar/NPM-Dataset/web/index.html')

# Wait for the user to manually click the 'Start' button and navigate to scene1.html
wait = WebDriverWait(driver, 60)

scenes = [5,7,10,12,15]
scenes_ordered = [1,2,3,4,5]
for i, j in zip(scenes, scenes_ordered):
    # Wait for the 'blocking-overlay' to be visible on scene{j}.html
    blocking_overlay = wait.until(EC.visibility_of_element_located((By.ID, 'blocking-overlay')))

    wait.until(EC.frame_to_be_available_and_switch_to_it((By.ID, 'myIframe')))

    time.sleep(13)

    ithor_box = driver.find_element(By.CLASS_NAME, "ant-checkbox-input")
    ithor_box.click()

    map = driver.find_elements(By.CLASS_NAME, "ant-card-cover")
    driver.execute_script("arguments[0].click();", map[i])


    driver.switch_to.default_content()


    next_button = WebDriverWait(driver, 300).until(EC.element_to_be_clickable((By.ID, 'next-button')))
    WebDriverWait(driver, 300).until(EC.visibility_of_element_located((By.ID, 'submission-message')))


    scene_data = extract_textbox_values(driver, f'Scene{j}')
    all_scene_data.append(scene_data)


df = pd.DataFrame(all_scene_data)
df.to_csv('scene_commands.csv', index=False)



# # Now that scene2.html is loaded, you can interact with its elements
# # Example: Wait for the 'blocking-overlay' to be visible on scene2.html
# blocking_overlay = wait.until(EC.visibility_of_element_located((By.ID, 'blocking-overlay')))

# wait.until(EC.frame_to_be_available_and_switch_to_it((By.ID, 'myIframe')))

# time.sleep(12.75)

# ithor_box = driver.find_element(By.CLASS_NAME, "ant-checkbox-input")
# ithor_box.click()

# # time.sleep(1)
# map = driver.find_elements(By.CLASS_NAME, "ant-card-cover")
# driver.execute_script("arguments[0].click();", map[7])



# # Now that scene3.html is loaded, you can interact with its elements
# # Example: Wait for the 'blocking-overlay' to be visible on scene3.html
# blocking_overlay = wait.until(EC.visibility_of_element_located((By.ID, 'blocking-overlay')))

# wait.until(EC.frame_to_be_available_and_switch_to_it((By.ID, 'myIframe')))

# time.sleep(12.25)

# ithor_box = driver.find_element(By.CLASS_NAME, "ant-checkbox-input")
# ithor_box.click()

# # time.sleep(1)
# map = driver.find_elements(By.CLASS_NAME, "ant-card-cover")
# driver.execute_script("arguments[0].click();", map[10])




# # Now that scene4.html is loaded, you can interact with its elements
# # Example: Wait for the 'blocking-overlay' to be visible on scene4.html
# blocking_overlay = wait.until(EC.visibility_of_element_located((By.ID, 'blocking-overlay')))

# wait.until(EC.frame_to_be_available_and_switch_to_it((By.ID, 'myIframe')))

# time.sleep(12.25)

# ithor_box = driver.find_element(By.CLASS_NAME, "ant-checkbox-input")
# ithor_box.click()

# # time.sleep(1)
# map = driver.find_elements(By.CLASS_NAME, "ant-card-cover")
# driver.execute_script("arguments[0].click();", map[12])



# # Now that scene5.html is loaded, you can interact with its elements
# # Example: Wait for the 'blocking-overlay' to be visible on scene5.html
# blocking_overlay = wait.until(EC.visibility_of_element_located((By.ID, 'blocking-overlay')))

# wait.until(EC.frame_to_be_available_and_switch_to_it((By.ID, 'myIframe')))

# time.sleep(12.25)

# ithor_box = driver.find_element(By.CLASS_NAME, "ant-checkbox-input")
# ithor_box.click()

# # time.sleep(1)
# map = driver.find_elements(By.CLASS_NAME, "ant-card-cover")
# driver.execute_script("arguments[0].click();", map[16])

breakpoint()


# Close the browser
driver.quit()