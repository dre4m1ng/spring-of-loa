import requests
import json

class API:
    def __init__(self, char_name, api_key):
        self.char_name = char_name
        self.headers = {
            'accept' : 'application/json',
            'authorization' : api_key
        }
    
    def profile(self):
        url = f'https://developer-lostark.game.onstove.com/armories/characters/{self.char_name}/profiles'
        response = requests.get(url, headers=self.headers)
        user_profile = response.json()
        return user_profile
    
    def equipment(self):
        url = f'https://developer-lostark.game.onstove.com/armories/characters/{self.char_name}/equipment'
        response = requests.get(url, headers=self.headers)
        user_equipment = response.json()
        return user_equipment