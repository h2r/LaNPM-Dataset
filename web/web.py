from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import time
import pandas as pd
import os
import drive_upload


# List to store data dictionaries for all scenes
all_scene_data = []
# Function to extract text box values and return them as a dictionary
def extract_textbox_values(driver, scene_name):
    data = {'Scene Name': scene_name}
    for i in range(1, 6):  # Assuming there are 5 command text boxes
        textbox = driver.find_element(By.ID, f'command{i}')
        data[f'Command {i}'] = textbox.get_attribute('value')
    return data


def check_selection(driver, element):
    driver.switch_to.frame(driver.find_element(By.ID, 'myIframe'))
    res = element.find_element(By.XPATH, '..')
    class_name = res.get_attribute('class')
    if '1kg7uqz' not in class_name:
        return False
    return True


chrome_options = Options()
chrome_options.add_experimental_option("detach", True)


driver = webdriver.Chrome(options=chrome_options)


driver.get('file:///home/ahmedjaafar/NPM-Dataset/web/index.html')
driver.implicitly_wait(1)

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


    # Before moving to the next scene, check the state
    # if not check_selection(driver, map[i]):
    #     # If state has changed, show an alert
    #     driver.execute_script("alert(\"You changed the simulator settings. Please don't do that. Refresh the page and start over\");")
    
    
    driver.switch_to.default_content()
    next_button = WebDriverWait(driver, 300).until(EC.element_to_be_clickable((By.ID, 'next-button')))
    if not check_selection(driver, map[i]):
        # If the check fails, display an alert and don't proceed to the next page
        driver.execute_script("alert(\"You changed the simulator settings. Please don't do that. Refresh the page and start over\");")
    else:
        WebDriverWait(driver, 300).until(EC.visibility_of_element_located((By.ID, 'submission-message')))

        scene_data = extract_textbox_values(driver, f'Scene{j}')
        all_scene_data.append(scene_data)


# Check if the 'user.txt' file exists in the current folder
if not os.path.exists('user.txt'):
    # If the file does not exist, create it and write the number 0
    with open('user.txt', 'w') as file:
        file.write('0')

# Read the current number from 'user.txt'
with open('user.txt', 'r') as file:
    number = int(file.read())

# Save the DataFrame as a CSV file with the name "commands_{number}.csv"
csv_filename = f'commands_participant{number}.csv'
df = pd.DataFrame(all_scene_data)
df.to_csv(csv_filename, index=False)

# Increment the number and update 'user.txt'
new_number = number + 1
with open('user.txt', 'w') as file:
    file.write(str(new_number))


#upload csv to google drive shared folder
service = drive_upload.service_account_login()
folder_id = '18sVFbyUGcmnRavVPgJTiW2Pa5D3gzubp'
file_path = csv_filename  
file_name = csv_filename

file_id = drive_upload.upload_file(service, file_path, file_name, folder_id)
print(f"Uploaded file ID: {file_id}")

if os.path.exists(file_name):
    os.remove(file_name)
    print(f"{file_name} has been deleted.")


breakpoint()

#idea:
#put a verify button
#verify button is disabled until boxes are filled
#next button is disabled until verified
#if verification fails, pop up alert, user refreshes (maybe make a refresh button in the alert)
#if verfication passes, save the data and disable/remove the iframe so they don't mess with it further
#once verification passes, enable next button so user can click and move to the next page