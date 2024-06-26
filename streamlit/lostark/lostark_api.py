import requests
import json

class API:
    def __init__(self, char_name, api_key):
        self.char_name = char_name
        self.headers = {
            'accept' : 'application/json',
            'authorization' : 'bearer ' + api_key
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
    
    def usr_info(self):
        user_lv = self.profile().get('ItemAvgLevel')
        user_equip = self.equipment()
        
        el_sum = 0
        chowal = 0
        if float(user_lv.replace(',', '')) >= 1600:
            for i in range(6):
                eq_json = json.loads(user_equip[i].get('Tooltip'))

                eq_dict = {}
                for key, value in eq_json.items():
                    eq_dict[key] = value

                if i != 0:
                    try:
                        el01_lv = int(eq_dict.get('Element_010').get('value').get('Element_000').get('contentStr').get('Element_000').get('contentStr').split('</FONT>')[1][-1])
                        el02_lv = int(eq_dict.get('Element_010').get('value').get('Element_000').get('contentStr').get('Element_001').get('contentStr').split('</FONT>')[1][-1])
                    except (AttributeError, ValueError):
                        try:
                            el01_lv = int(eq_dict.get('Element_009').get('value').get('Element_000').get('contentStr').get('Element_000').get('contentStr').split('</FONT>')[1][-1])
                            el02_lv = int(eq_dict.get('Element_009').get('value').get('Element_000').get('contentStr').get('Element_001').get('contentStr').split('</FONT>')[1][-1])
                        except (AttributeError, ValueError):
                            try:
                                el01_lv = int(eq_dict.get('Element_008').get('value').get('Element_000').get('contentStr').get('Element_000').get('contentStr').split('</FONT>')[1][-1])
                                el02_lv = int(eq_dict.get('Element_008').get('value').get('Element_000').get('contentStr').get('Element_001').get('contentStr').split('</FONT>')[1][-1])
                            except AttributeError:
                                pass
                    el_sum += el01_lv + el02_lv

                if float(user_lv.replace(',', '')) >= 1620 and i == 1:
                    try:
                        chowal = int(eq_dict.get('Element_009').get('value').get('Element_000').get('contentStr').get('Element_001').get('contentStr').split('</img>')[-1].split('개')[0])
                    except:
                        try:
                            chowal = int(eq_dict.get('Element_008').get('value').get('Element_000').get('contentStr').get('Element_001').get('contentStr').split('</img>')[-1].split('개')[0])
                        except:
                            pass
                        
        return user_lv, el_sum, chowal
    