import string
import sys
from datetime import datetime
import execjs
import requests
import base64
import json
import threading
import queue
import uuid
import ctypes
import os
import time
import colr
from colorama import Fore as b
import datetime as dt
from typing import Dict, Optional
import tls_client
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import padding
from colorama import Fore as f
from threading import Lock
from fastapi import FastAPI, HTTPException, Request, Header
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import traceback
import hashlib
import logging
import random
from concurrent.futures import ThreadPoolExecutor
import binascii
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from javascript import require
import contextlib
import struct
from io import BytesIO
import secrets


jsdom = require('jsdom')
create_script = require("vm").Script
logging.getLogger('werkzeug').setLevel(logging.ERROR)
with open("webgl.json") as file:
    webgls = json.loads(file.read())

with open("arkose.js") as file:
    gctx = execjs.compile(file.read())


class Utils:
    solved = 0
    fail = 0
    spm = 0
    errors = 0
    supc = 0
    xxsupc = 0

    @contextlib.contextmanager
    def suppress_output(self):
        with open(os.devnull, 'w') as devnull:
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            try:
                sys.stdout = devnull
                sys.stderr = devnull
                yield
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr

    @staticmethod
    def generate_user_agent():
        chrome_version = str(random.randint(100, 129))
        return {
            'user_agent': f'Mozilla/5.0 (Windows NT 10.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{chrome_version}.0.0.0 Safari/537.36',
            'chrome_version': chrome_version
        }

    @staticmethod
    def find(data, value) -> str:
        for item in data:
            if item["key"] == value:
                return item["value"]

    @staticmethod
    def newrelic_time() -> str:
        a, b = str(time.time()).split(".")
        return str(a + b[0:5])

    @staticmethod
    def hex(data: str) -> str:
        return ''.join(f'{byte:02x}' for byte in data)

    @staticmethod
    def convert_salt(words: list, sig_bytes: list) -> list:
        salt = b''
        for word in words:
            salt += struct.pack('>I', word & 0xFFFFFFFF)
        return salt[:sig_bytes]

    @staticmethod
    def randsalt(ctx) -> list:
        return ctx.call(
            'randsigbyte',
            8,
        )

    @staticmethod
    def int_to_bytes(n: str, length: int) -> bytes:
        return n.to_bytes(length, byteorder='big', signed=True)

    @staticmethod
    def to_sigbytes(words: list, sigBytes: int) -> list:
        result = b''.join(Utils.int_to_bytes(word, 4) for word in words)
        return result[:sigBytes]

    @staticmethod
    def bytes_to_buffer(data: bytes) -> list:
        buffer = BytesIO(data)
        buffer.seek(0)
        content = buffer.read()
        return list(content)

    @staticmethod
    def random_integer(value: int) -> int:
        max_random_value = (2 ** 32 // value) * value
        while True:
            a = secrets.randbelow(2 ** 32)
            if a < max_random_value:
                return a % value

    @staticmethod
    def dict_to_list(data: dict) -> list:
        result = []
        for obj in data:
            result.append(data[obj])

        return result

    @staticmethod
    def uint8_array(size: int) -> list:
        v = bytearray(size)
        for i in range(len(v)):
            v[i] = Utils.random_integer(256)
        return Utils.bytes_to_buffer(v)

    @staticmethod
    def convert_key_to_sigbytes_format(key: bytes) -> list:
        key_words = []
        for i in range(0, len(key), 4):
            word = struct.unpack('>i', key[i:i + 4])[0]
            key_words.append(word)
        return key_words

    @staticmethod
    def get_coords(num: int) -> tuple:
        map = {
            1: [0, 0, 100, 100],
            2: [100, 0, 200, 100],
            3: [200, 0, 300, 100],
            4: [0, 100, 100, 200],
            5: [100, 100, 200, 200],
            6: [200, 100, 300, 200]
        }

        x1, y1, x2, y2 = map[num]
        x = (x1 + x2) / 2
        y = (y1 + y2) / 2

        if random.randint(0, 1):
            x += random.uniform(10.00001, 45.99999)
        else:
            x -= random.uniform(10.00001, 45.99999)

        if random.randint(0, 1):
            y += random.randint(0, 45)
        else:
            y -= random.randint(0, 45)

        return (float(x), int(y))

    @staticmethod
    def grid_answer_dict(answer):
        x, y = Utils.get_coords(answer)

        return {
            "px": str(round((x / 300), 2)),
            "py": str(round((y / 200), 2)),
            "x": x,
            "y": y
        }



class logger:
    colors_table = {
        "green": "#65fb07",
        "red": "#Fb0707",
        "yellow": "#c4bc18",
        "magenta": "#b207f5",
        "blue": "#001aff",
        "cyan": "#07baf5",
        "gray": "#3a3d40",
        "white": "#ffffff",
        "pink": "#c203fc",
        "light_blue": "#07f0ec",
        "orange": "#FFA200",
        "purple": "#4500d0"
    }

    @staticmethod
    def convert(color):
        if not color.__contains__("#"):
            return logger.colors_table[color]
        else:
            return color

    @staticmethod
    def color(c, obj):
        return colr.Colr().hex(logger.convert(c), obj)

    def print(*args: str) -> None:
        current_time = dt.datetime.now()
        date = current_time.strftime(f'%H:%M:%S')
        print(
            f"{logger.color('white', date)} {f.BLUE}| {b.BLUE}{logger.color('purple', 'Funcaptcha')}{b.RESET} | {' | '.join(args)}".replace(
                "|", f"{f.LIGHTBLACK_EX}|{f.WHITE}"))


class solver_stats:
    @staticmethod
    def calc_cpm():

        while True:
            try:
                before = Utils.solved
                time.sleep(15)
                after = Utils.solved
                Utils.spm = (after - before) * 4
            except Exception as e:
                print(f"Error in calc_cpm: {e}")
                Utils.spm = 0

    @staticmethod
    def rate(part, total):
        try:
            return str(round(part / total * 100, 4)) if total > 0 else "0.0000"
        except Exception as e:
            print(f"Error in rate calculation: {e}")
            return "unknown"

    @staticmethod
    def title():

        import platform
        is_windows = platform.system() == "Windows"

        def clear_screen():
            if is_windows:
                os.system("cls")
            else:
                os.system("clear")

        clear_screen()

        second = 0
        minute = 0
        hours = 0
        days = 0

        while True:
            second += 1
            if second == 60:
                second = 0
                minute += 1
            if minute == 60:
                minute = 0
                hours += 1
            if hours == 24:
                hours = 0
                days += 1

            elapsed = f"{str(days).zfill(2)}:{str(hours).zfill(2)}:{str(minute).zfill(2)}:{str(second).zfill(2)}"

            if is_windows:
                try:
                    ctypes.windll.kernel32.SetConsoleTitleW(
                        f"Funcaptcha | solved: {str(Utils.solved)} | fail: {str(Utils.fail)} | spm: {str(Utils.spm)} | sup: {solver_stats.rate(Utils.supc, Utils.supc + Utils.xxsupc)}% | suc: {solver_stats.rate(Utils.solved, Utils.solved + Utils.fail)}% | errors: {Utils.errors} | elapsed: {elapsed}"
                    )
                except Exception as e:
                    print(f"Failed to set console title: {e}")

            else:
                print(
                    f"Elapsed: {elapsed} | solved: {Utils.solved} | fail: {Utils.fail} | spm: {Utils.spm} | sup: {solver_stats.rate(Utils.supc, Utils.supc + Utils.xxsupc)}% | suc: {solver_stats.rate(Utils.solved, Utils.solved + Utils.fail)}% | errors: {Utils.errors}",
                    end="\r",
                )

            time.sleep(1)


class Arkose:
    @staticmethod
    def decrypt_data(data, main):
        ciphertext = base64.b64decode(data['ct'])
        iv_bytes = binascii.unhexlify(data['iv'])
        salt_bytes = binascii.unhexlify(data['s'])
        salt_words = Arkose.from_sigbytes(salt_bytes)
        key_words = Arkose.generate_other_key(main, salt_words)
        key_bytes = Utils.to_sigbytes(key_words, 32)
        iv_bytes = Utils.to_sigbytes(key_words[-4:], 16)
        cipher = AES.new(key_bytes, AES.MODE_CBC, iv_bytes)
        plaintext = unpad(cipher.decrypt(ciphertext), AES.block_size)
        return plaintext

    @staticmethod
    def from_sigbytes(sigBytes: bytes) -> list:
        padded_length = (len(sigBytes) + 3) // 4 * 4
        padded_bytes = sigBytes.ljust(padded_length, b'\0')

        words = [int.from_bytes(padded_bytes[i:i + 4], byteorder='big') for i in range(0, len(padded_bytes), 4)]
        return words

    @staticmethod
    def encrypt_double(self, main: str, extra: str) -> str:
        salt_words = Utils.randsalt(gctx)
        key_words = Arkose.generate_other_key(main, salt_words)
        key_bytes = Utils.to_sigbytes(key_words, 32)
        iv_bytes = Utils.to_sigbytes(key_words[-4:], 16)
        salt_bytes = Utils.to_sigbytes(salt_words, 8)
        cipher = AES.new(key_bytes, AES.MODE_CBC, iv_bytes)
        ciphertext = cipher.encrypt(pad(extra.encode('utf-8'), AES.block_size))

        return json.dumps({
            "ct": base64.b64encode(ciphertext).decode(),
            "iv": Utils.hex(iv_bytes),
            "s": Utils.hex(salt_bytes)
        }).replace(" ", "")

    def is_flagged(data):
        if not data or not isinstance(data, list):
            return False
        values = [value for d in data for value in d.values()]
        if not values:
            return False

        def ends_with_uppercase(value):
            return value and value[-1] in string.ascii_uppercase

        return all(ends_with_uppercase(value) for value in values)

    @staticmethod
    def t_guess(self, session_token: str, guesses: list, dapibCode: str) -> str:
        sess, ion = session_token.split(".")
        answers = []
        for guess in guesses:
            if "index" in str(guesses):
                answers.append({"index": json.loads(guess)["index"], sess: ion, })
            else:
                guess = json.loads(guess)
                answers.append({"px": guess['px'], "py": guess['py'], "x": guess['x'], "y": guess['y'], sess: ion})

        resource_loader = jsdom.ResourceLoader({
            "userAgent": self.useragent
        })
        vm = jsdom.JSDOM("", {
            "runScripts": "dangerously",
            "resources": resource_loader,
            "pretendToBeVisual": True,
            "storageQuota": 10000000

        }).getInternalVMContext()

        create_script("""
        response=null;

        window.parent.ae={"answer":answers}

        window.parent.ae[("dapibRecei" + "ve")]=function(data) {
        response=JSON.stringify(data);
        }
        """.replace("answers", json.dumps(answers).replace('"index"', 'index'))).runInContext(vm)

        create_script(dapibCode).runInContext(vm)
        result = json.loads(create_script("response").runInContext(vm))

        if Arkose.is_flagged(result["tanswer"]):
            for array in result["tanswer"]:
                for item in array:
                    array[item] = array[item][:-1]

        return Arkose.encrypt_double(self, session_token, json.dumps(result["tanswer"]).replace(" ", ""))

    @staticmethod
    def generate_key_(password: str, salt: bytes, key_size: int, iterations: int) -> bytes:
        hasher = hashlib.md5()
        key = b''
        block = None

        while len(key) < key_size:
            if block:
                hasher.update(block)
            hasher.update(password.encode())
            hasher.update(salt)
            block = hasher.digest()
            hasher = hashlib.md5()

            for _ in range(1, iterations):
                hasher.update(block)
                block = hasher.digest()
                hasher = hashlib.md5()

            key += block

        return key[:key_size]

    @staticmethod
    def generate_other_key(data: str, salt: list) -> list:
        sig_bytes = 8
        key_size = 48
        iterations = 1
        salt = Utils.convert_salt(salt, sig_bytes)
        key = Arkose.generate_key_(data, salt, key_size, iterations)

        return Utils.convert_key_to_sigbytes_format(key)

    @staticmethod
    def encrypt_ct(text: bytes, key: bytes, iv: bytes) -> bytes:
        encryptor = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend()).encryptor()
        padder = padding.PKCS7(algorithms.AES.block_size).padder()
        padded_plain_text = padder.update(text) + padder.finalize()
        cipher_text = encryptor.update(padded_plain_text) + encryptor.finalize()
        return cipher_text

    @staticmethod
    def generate_key(ctx: execjs.compile, s_value: str, useragent: str) -> list:
        key = Utils.dict_to_list(ctx.call(
            'genkey',
            useragent,
            s_value
        ))
        return key

    @staticmethod
    def make_encrypted_dict(self, data: str) -> str:
        s_value = Utils.hex(Utils.uint8_array(8))
        iv_value = Utils.uint8_array(16)
        key = Arkose.generate_key(
            gctx,
            s_value,
            f"{self.useragent}{self.x_ark_value}"
        )

        result = Arkose.encrypt_ct(
            text=bytes(data.encode()),
            key=bytes(key),
            iv=bytes(iv_value)
        )

        return json.dumps({
            "ct": base64.b64encode(result).decode(),
            "s": s_value,
            "iv": Utils.hex(iv_value)
        }).replace(" ", "")

    @staticmethod
    def x_ark_value() -> str:
        now = int(time.time())
        return str(now - (now % 21600))

    @staticmethod
    def generate_pt():
        start_time = time.time()
        time.sleep(random.uniform(0.04, 0.07))
        end_time = time.time()

        pt = end_time - start_time
        return pt

    @staticmethod
    def generate_aht(num_trials=random.randint(1, 7)):
        total_time = 0

        for _ in range(num_trials):
            pt = Arkose.generate_pt()
            total_time += pt

        aht = total_time / num_trials
        return aht

    @staticmethod
    def generate_cs():
        return uuid.uuid4().hex[:10]

    @staticmethod
    def generate_g():
        return uuid.uuid4().hex[:12]

    @staticmethod
    def generate_h(cs, g):
        data_to_hash = cs + g
        h = hashlib.sha256(data_to_hash.encode()).hexdigest()
        return h




class Funcaptcha:
    def __init__(
            self,
            apiurl: str,
            siteurl: str,
            sitekey: str,
            data: dict,
            blob: str,
            custom_cookies: dict,
            custom_locale: str,
            proxy: Optional[str] = None,
    ) -> None:

        ua_data = Utils.generate_user_agent()
        print(ua_data)
        self.useragent = ua_data['user_agent']
        self.siteurl = siteurl
        self.sitekey = sitekey
        self.apiurl = apiurl
        self.blob = blob

        self.base_headers = {
            "User-Agent": self.useragent,
        }

        self.session = tls_client.Session(
            client_identifier="chrome_120",
            random_tls_extension_order=True
        )
        if custom_cookies:
            self.session.cookies.update(custom_cookies)

        self.available_locales = [
            "en-IN", "en-US",
        ]
        if custom_locale:
            self.locale = custom_locale
        else:
            self.locale = random.choice(self.available_locales)

        if proxy:
            self.session.proxies = {"http": proxy, "https": proxy}
            self.session.trust_env = False

        self.x_ark_value = Arkose.x_ark_value()

        captchadata = self.session.get(f"{self.apiurl}/v2/{self.sitekey}/api.js", headers={
            'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'accept-language': f'en-US,en;q=0.9,{self.locale};q=0.8,{self.locale.split("-")[0]};q=0.7',
            'upgrade-insecure-requests': '1',
            **self.base_headers,
        }).text.split("/enforcement.")

        self.capi_version = captchadata[0].split('"')[-1]
        self.enforcement_hash = captchadata[1].split('.html')[0]
        self.window__ancestor_origins = data["window__ancestor_origins"]
        self.window__tree_index = data["window__tree_index"]
        self.client_config__sitedata_location_href = data["client_config__sitedata_location_href"]
        self.window__tree_structure = data["window__tree_structure"]

        self.session.headers = {
            "Accept-Encoding": "gzip, deflate, br, zstd",
            "Accept-Language": f"{self.locale},{self.locale.split('-')[0]};q=0.9,en-US;q=0.8,en;q=0.7",
            "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
            "Host": self.apiurl.split('https://')[1],
            "Referer": f"{self.apiurl}/v2/{self.capi_version}/enforcement.{self.enforcement_hash}.html",
            "x-ark-esync-value": self.x_ark_value,
            **self.base_headers,
        }

    def make_mm(self):
        x = random.randint(100, 150)
        y = random.randint(30, 90)

        return base64.b64encode(json.dumps({
            "mbio": f"38{random.randint(10, 70)},0,{x},{y};38{random.randint(80, 90)},1,{x},{y};38{random.randint(92, 99)},2,{x},{y};",
            "tbio": f"34{random.randint(10, 99)},0,{x},{y};",
            "kbio": ""
        }).replace(" ", "").encode()).decode()

    def answer(self, result, answers, i):
        data = {
            'session_token': self.session_token,
            'game_token': self.gameid,
            'sid': self.r_continent,
            'guess': result,
            'render_type': 'canvas',
            'analytics_tier': '40',
            'bio': self.make_mm(),
            'is_compatibility_mode': 'false',
        }

        if self.dapibCode:
            data['tguess'] = Arkose.t_guess(self, self.session_token, answers, self.dapibCode)

        response = self.session.post(f'{self.apiurl}/fc/ca/', data=data, headers={
            "Accept-Language": f"{self.locale},{self.locale.split('-')[0]};q=0.9,en-US;q=0.8,en;q=0.7",
            "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
            "Host": self.apiurl.split("https://")[1],
            "Origin": self.apiurl,
            "Referer": f"{self.apiurl}/fc/assets/ec-game-core/game-core/1.22.0/standard/index.html?session={self.token}&r={self.r_continent}&meta=3&meta_width=300&metabgclr=transparent&metaiconclr=%23555555&guitextcolor=%23000000&pk={self.sitekey}&dc=1&at=40&ag=101&cdn_url=https%3A%2F%2F{self.apiurl.split('https://')[1]}%2Fcdn%2Ffc&lurl=https%3A%2F%2Faudio-{self.r_continent}.arkoselabs.com&surl=https%3A%2F%2F{self.apiurl.split('https://')[1]}&smurl=https%3A%2F%2F{self.apiurl.split('https://')[1]}%2Fcdn%2Ffc%2Fassets%2Fstyle-manager&theme=default",
            "Sec-Fetch-Site": "same-origin",
            "X-NewRelic-Timestamp": Utils.newrelic_time(),
            "X-Requested-ID": Arkose.encrypt_double(self, f"REQUESTED{self.token}ID", json.dumps({"sc": [
                random.randint(100, 300),
                random.randint(100, 300)
            ]}).replace(" ", "")),
            "X-Requested-With": "XMLHttpRequest",
            **self.base_headers,
        })
        return response.json()

    def callback(self, data):
        self.session.post(f"{self.apiurl}/fc/a/", data=data, headers={
            "Accept-Language": f"{self.locale},{self.locale.split('-')[0]};q=0.9,en-US;q=0.8,en;q=0.7",
            "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
            "Origin": self.apiurl,
            "Referer": f"{self.apiurl}/fc/assets/ec-game-core/game-core/1.22.0/standard/index.html?session={self.token}&r={self.r_continent}&meta=7&meta_height=325&metabgclr=%23ffffff&metaiconclr=%23757575&mainbgclr=%23ffffff&maintxtclr=%231B1B1B&guitextcolor=%23747474&lang={self.locale.split('-')[0]}&pk={self.sitekey}&at=40&ag=101&cdn_url=https%3A%2F%2F{self.apiurl.split('https://')[1]}%2Fcdn%2Ffc&lurl=https%3A%2F%2Faudio-{self.r_continent}.arkoselabs.com&surl=https%3A%2F%2F{self.apiurl.split('https://')[1]}&smurl=https%3A%2F%2F{self.apiurl.split('https://')[1]}%2Fcdn%2Ffc%2Fassets%2Fstyle-manager&theme=default",
            "Sec-Fetch-Site": "same-origin",
            "X-NewRelic-Timestamp": Utils.newrelic_time(),
            "X-Requested-With": "XMLHttpRequest",
            **self.base_headers,
        })

    def solve(self):
        try:
            self.solve_time = time.time()
            challenge_data = self._generate_challenge()
            if "DENIED ACCESS" in str(challenge_data):
                return {"success": False, "err": "invalid blob", "token": None}

            del self.session.headers["Content-Type"]
            del self.session.headers["x-ark-esync-value"]
            self.token = challenge_data["token"].split("|")[0]
            logger.print("Token:", f.LIGHTGREEN_EX + self.token)
            self.r_continent = challenge_data["token"].split("|r=")[1].split("|")[0]
            self.session.get(f'{self.apiurl}/fc/gc/', params={'token': self.token})

            self.session.headers.update({
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
                "Sec-Fetch-Dest": "iframe",
                "Sec-Fetch-Mode": "navigate",
                "Upgrade-Insecure-Requests": "1"
            })
            referer = self.session.get(
                f'{self.apiurl}/fc/assets/ec-game-core/game-core/1.22.0/standard/index.html',
                params={
                    'session': self.token,
                    'r': self.r_continent,
                    'meta': '3',
                    'meta_width': '300',
                    'metabgclr': 'transparent',
                    'metaiconclr': '#555555',
                    'guitextcolor': '#000000',
                    'pk': self.sitekey,
                    'dc': '1',
                    'at': '40',
                    'ag': '101',
                    'cdn_url': f'{self.apiurl}/cdn/fc',
                    'lurl': f'https://audio-{self.r_continent}.arkoselabs.com',
                    'surl': self.apiurl,
                    'smurl': f'{self.apiurl}/cdn/fc/assets/style-manager',
                    'theme': 'default'
                }).url
            self.callback(
                {"sid": self.r_continent, "session_token": self.token, "analytics_tier": "40", "disableCookies": "true",
                 "render_type": "canvas", "is_compatibility_mode": "false", "category": "Site URL",
                 "action": f"{self.apiurl}/v2/{self.capi_version}/enforcement.{self.enforcement_hash}.html"})

            if "sup=1|" in challenge_data["token"]:
                self.session.headers.update({
                    "Sec-Fetch-Dest": "script",
                    "Sec-Fetch-Mode": "no-cors",
                    "Sec-Fetch-Site": "same-origin"
                })

                self.session.get(
                    f"{self.apiurl}/fc/a/",
                    params={
                        "callback": f"__jsonp_{str(int(time.time() * 1000))}",
                        "category": "loaded",
                        "action": "game loaded",
                        "session_token": self.token,
                        "data[public_key]": self.sitekey,
                        "data[site]": self.siteurl
                    }
                )
                logger.print("Suppressed", "Waves: 0", f.LIGHTGREEN_EX + self.token)
                Utils.solved += 1
                Utils.supc += 1

                return {"success": True, "err": None, "token": challenge_data["token"],
                        "procces_time": time.time() - self.solve_time}

            Utils.xxsupc += 1
            self.session.headers.update({
                "referer": referer,
                "content-type": "application/x-www-form-urlencoded; charset=UTF-8",
                "origin": self.apiurl,
                "x-newrelic-timestamp": Utils.newrelic_time(),
                "x-requested-with": "XMLHttpRequest",
            })

            result = self._task_data()

            del self.session.headers["content-type"]
            del self.session.headers["origin"]
            del self.session.headers["x-newrelic-timestamp"]
            del self.session.headers["x-requested-with"]

            self.gameid = result["challengeID"]
            self.session_token = result["session_token"]
            self.callback(
                {"sid": self.r_continent, "session_token": self.token, "analytics_tier": "40", "disableCookies": "true",
                 "game_token": self.gameid, "game_type": "4", "render_type": "canvas", "is_compatibility_mode": "false",
                 "category": "loaded", "action": "game loaded"})

            try:
                game = result["game_data"]["instruction_string"]
            except:
                game = result["game_data"]["game_variant"]

            waves = str(result["game_data"]["waves"])
            logger.print(f"Type: {game} Quantity: {waves}")
            if int(waves) >= 20:
                logger.print(game, ':', str(waves))
                return {"success": False, "err": "too many waves"}

            if result.get("dapib_url"):
                self.dapibCode = self.session.get(result["dapib_url"]).text
                self.session.get(
                    f"{self.apiurl}/params/sri/dapib/{result['dapib_url'].split(f'{self.r_continent}/')[1].split('.js')[0]}")

            else:
                self.dapibCode = 0

            cs = Arkose.generate_cs()
            g = Arkose.generate_g()
            self.callback(
                {"sid": self.r_continent, "session_token": self.token, "analytics_tier": "40", "disableCookies": "true",
                 "game_token": self.gameid, "game_type": "4", "render_type": "canvas", "is_compatibility_mode": "false",
                 "category": "begin app", "action": "user clicked verify", "cs_": cs,
                 "ct_": str(random.randint(10, 99)), "g_": g, "h_": Arkose.generate_h(cs, g),
                 "pt_": Arkose.generate_pt(), "aht_": Arkose.generate_aht()})

            image_encryption_enabled = result["game_data"]["customGUI"].get('encrypted_mode')
            if image_encryption_enabled:
                self.image_decryption_key = self.session.post(f'{self.apiurl}/fc/ekey/', data={
                    'session_token': self.session_token,
                    'game_token': self.gameid,
                    'sid': self.r_continent
                }).json()['decryption_key']

            answers = []
            test = []
            total_images = len(result["game_data"]["customGUI"]["_challenge_imgs"])
            logger.print(f"Type: {game} Number: {waves}")
            logger.print(f"Total number of images to be processed: {total_images}")

            for img_index, img in enumerate(result["game_data"]["customGUI"]["_challenge_imgs"]):
                if image_encryption_enabled:
                    base64_img = Arkose.decrypt_data(self.session.get(img).json(), self.image_decryption_key).decode()
                else:
                    val = self.session.cookies.get('_cfuvid')
                    self.session.headers.update({
                        "Cookie": f"_cfuvid={val}"
                    })

                    attempt = 0
                    max_attempts = 10

                    while attempt < max_attempts:
                        try:
                            response = self.session.get(img)
                            if response.status_code == 200:
                                base64_img = base64.b64encode(response.content).decode('utf-8')
                                break
                            elif response.status_code == 403:
                                logger.print(f"Attempt {attempt + 1} failed with 403. Retrying...")
                            else:
                                logger.print(
                                    f"Attempt {attempt + 1} failed with status code {response.status_code}. Retrying...")

                        except requests.exceptions.RequestException as e:
                            logger.print(f"Attempt {attempt + 1} failed due to exception: {e}")

                        attempt += 1
                        time.sleep(1)

                    if attempt == max_attempts:
                        raise Exception(f"Failed to fetch image after {max_attempts} attempts")

                    if not os.path.exists("jpg"):
                        os.makedirs("jpg")

                    with open(f"jpg/{img_index}.jpg", "wb") as file:
                        file.write(response.content)

                    predicted_index = input(f"Please enter the index of the image {img_index + 1}:")

                    index = predicted_index
                    logger.print(f"Image {img_index + 1} index: {index}")
                    if result["game_data"]["gameType"] == 3:
                        index += 1

                    test.append(str(index))

                    if result["game_data"]["gameType"] == 4:
                        answers.append(json.dumps({"index": index}).replace(" ", ""))
                    elif result["game_data"]["gameType"] == 3:
                        answers.append(json.dumps(Utils.grid_answer_dict(index)).replace(" ", ""))

            logger.print(f"All images processed, submitting answer: [{','.join(test)}]")

            if len(answers) != total_images:
                logger.print(f"Error: Got {len(answers)} answers, expected {total_images}")
                return {"success": False, "err": "incomplete answers", "token": None}

            solveresult = self.answer(
                result=Arkose.encrypt_double(self, self.session_token, str([",".join(answers)]).replace("'", "")),
                answers=answers,
                i=index
            )

            if image_encryption_enabled:
                self.image_decryption_key = solveresult.get('decryption_key')

            if solveresult["solved"]:
                logger.print(
                    game,
                    f"Waves: {waves}",
                    f"Answers: {str(len(test))}:[{','.join(test)}]",
                    f.LIGHTGREEN_EX + self.token
                )
                Utils.solved += 1
                return {
                    "game": game,
                    "waves": waves,
                    "success": True,
                    "err": None,
                    "token": challenge_data["token"],
                    "procces_time": time.time() - self.solve_time,
                }
            else:
                logger.print(
                    game,
                    f"Waves: {waves}",
                    f"Answers: {str(len(test))}:[{','.join(test)}]",
                    f.RED + "Failed_to_solve."
                )
                Utils.fail += 1
                return {
                    "game": game,
                    "waves": waves,
                    "success": False,
                    "err": "ai_fail",
                    "token": None,
                }



        except Exception as err:
            logger.print(traceback.format_exc())
            Utils.errors += 1
            return {"success": False, "err": "internal error", "token": None}

    def _task_data(self):
        return self.session.post(f"{self.apiurl}/fc/gfct/", data={
            'token': self.token,
            'sid': self.r_continent,
            'render_type': 'canvas',
            'lang': '',
            'isAudioGame': 'false',
            'is_compatibility_mode': 'false',
            'apiBreakerVersion': 'green',
            'analytics_tier': '40',
        }).json()

    def md5_hash(self, data):
        md5_hash = hashlib.md5()
        md5_hash.update(data.encode('utf-8'))
        return md5_hash.hexdigest()

    def process_fp(self, fpdata):
        result = []
        for item in fpdata:
            result.append(item.split(":")[1])
        return ';'.join(result)

    def proccess_webgl2(self, data):
        result = []

        for item in data:
            result.append(item["key"])
            result.append(item["value"])

        return ','.join(result) + ',webgl_hash_webgl,'

    def find(self, data, key):
        for item in data:
            if item["key"] == key:
                return item["value"]

    def random_pixel_depth(self):
        pixel_depths = [24, 30]
        return random.choice(pixel_depths)

    def bda(self):
        time_now = time.time()

        resolutions = [
            (3440, 1440, 3440, 1400),
            (1924, 1007, 1924, 1007),
            (1920, 1080, 1920, 1040),
            (1280, 720, 1280, 672),
            (1920, 1080, 1920, 1032),
            (1366, 651, 1366, 651),
            (1366, 768, 1366, 738),
            (1920, 1080, 1920, 1050)
        ]
        height, width, awidth, aheight = random.choice(resolutions)

        webgl_data = random.choice(webgls)

        self.cfp = webgl_data["fe"]["CFP"]

        fp1 = [
            "DNT:unknown",
            f"L:{self.locale}",
            "D:24",
            "PR:1",
            f"S:{height},{width}",
            f"AS:{awidth},{aheight}",
            "TO:480",
            "SS:true",
            "LS:true",
            "IDB:true",
            "B:false",
            "ODB:false",
            "CPUC:unknown",
            "PK:Win32",
            f"CFP:{self.cfp}",
            "FR:false",
            "FOS:false",
            "FB:false",
            "JSF:Arial,Arial Black,Arial Narrow,Calibri,Cambria,Cambria Math,Comic Sans MS,Consolas,Courier,Courier New,Georgia,Helvetica,Impact,Lucida Console,Lucida Sans Unicode,Microsoft Sans Serif,MS Gothic,MS PGothic,MS Sans Serif,MS Serif,Palatino Linotype,Segoe Print,Segoe Script,Segoe UI,Segoe UI Light,Segoe UI Semibold,Segoe UI Symbol,Tahoma,Times,Times New Roman,Trebuchet MS,Verdana,Wingdings",
            "P:Chrome PDF Viewer,Chromium PDF Viewer,Microsoft Edge PDF Viewer,PDF Viewer,WebKit built-in PDF",
            "T:0,false,false",
            f"H:{random.randint(2, 20)}",
            "SWF:false"
        ]

        webgl = random.choice(random.choice(webgls)["enhanced_fp"])
        enhanced_fp = [
            {
                "key": "webgl_extensions",
                "value": webgl["webgl_extensions"]
            },
            {
                "key": "webgl_extensions_hash",
                "value": webgl["webgl_extensions_hash"]
            },
            {
                "key": "webgl_renderer",
                "value": webgl["webgl_renderer"]
            },
            {
                "key": "webgl_vendor",
                "value": webgl["webgl_vendor"]
            },
            {
                "key": "webgl_version",
                "value": webgl["webgl_version"]
            },
            {
                "key": "webgl_shading_language_version",
                "value": webgl["webgl_shading_language_version"]
            },
            {
                "key": "webgl_aliased_line_width_range",
                "value": webgl["webgl_aliased_line_width_range"]
            },
            {
                "key": "webgl_aliased_point_size_range",
                "value": webgl["webgl_aliased_point_size_range"]
            },
            {
                "key": "webgl_antialiasing",
                "value": "yes"
            },
            {
                "key": "webgl_bits",
                "value": webgl["webgl_bits"]
            },
            {
                "key": "webgl_max_params",
                "value": webgl["webgl_max_params"]
            },
            {
                "key": "webgl_max_viewport_dims",
                "value": webgl["webgl_max_viewport_dims"]
            },
            {
                "key": "webgl_unmasked_vendor",
                "value": webgl["webgl_unmasked_vendor"]
            },
            {
                "key": "webgl_unmasked_renderer",
                "value": webgl["webgl_unmasked_renderer"]
            },
            {
                "key": "webgl_vsf_params",
                "value": webgl["webgl_vsf_params"]
            },
            {
                "key": "webgl_vsi_params",
                "value": webgl["webgl_vsi_params"]
            },
            {
                "key": "webgl_fsf_params",
                "value": webgl["webgl_fsf_params"]
            },
            {
                "key": "webgl_fsi_params",
                "value": webgl["webgl_fsi_params"]
            }]

        enhanced_fp.append({
            "key": "webgl_hash_webgl",
            "value": gctx.call("x64hash128", self.proccess_webgl2(enhanced_fp))
        })

        enhanced_fp_more = [
            {
                "key": "user_agent_data_brands",
                "value": "Chromium,Not;A=Brand,Google Chrome"
            },
            {
                "key": "user_agent_data_mobile",
                "value": False
            },
            {
                "key": "navigator_connection_downlink",
                "value": 10
            },
            {
                "key": "navigator_connection_downlink_max",
                "value": None
            },
            {
                "key": "network_info_rtt",
                "value": 50
            },
            {
                "key": "network_info_save_data",
                "value": False
            },
            {
                "key": "network_info_rtt_type",
                "value": None
            },
            {
                "key": "screen_pixel_depth",
                "value": self.random_pixel_depth()
            },
            {
                "key": "navigator_device_memory",
                "value": 8
            },
            {
                "key": "navigator_pdf_viewer_enabled",
                "value": True
            },
            {
                "key": "navigator_languages",
                "value": "en-US,en"
            },
            {
                "key": "window_inner_width",
                "value": 0
            },
            {
                "key": "window_inner_height",
                "value": 0
            },
            {
                "key": "window_outer_width",
                "value": int(width)
            },
            {
                "key": "window_outer_height",
                "value": int(height)
            },
            {
                "key": "browser_detection_firefox",
                "value": False
            },
            {
                "key": "browser_detection_brave",
                "value": False
            },
            {
                "key": "browser_api_checks",
                "value": [
                    "permission_status: true",
                    "eye_dropper: true",
                    "audio_data: true",
                    "writable_stream: true",
                    "css_style_rule: true",
                    "navigator_ua: true",
                    "barcode_detector: false",
                    "display_names: true",
                    "contacts_manager: false",
                    "svg_discard_element: false",
                    "usb: defined",
                    "media_device: defined",
                    "playback_quality: true"
                ]
            },
            {
                "key": "browser_object_checks",
                "value": self.md5_hash("chrome")
            },
            {
                "key": "29s83ih9",
                "value": self.md5_hash("false") + "⁣"
            },
            {
                "key": "audio_codecs",
                "value": "{\"ogg\":\"probably\",\"mp3\":\"probably\",\"wav\":\"probably\",\"m4a\":\"maybe\",\"aac\":\"probably\"}"
            },
            {
                "key": "audio_codecs_extended_hash",
                "value": self.md5_hash(
                    '{"audio/mp4; codecs=\\"mp4a.40\\"":{"canPlay":"maybe","mediaSource":false},"audio/mp4; codecs=\\"mp4a.40.1\\"":{"canPlay":"","mediaSource":false},"audio/mp4; codecs=\\"mp4a.40.2\\"":{"canPlay":"probably","mediaSource":true},"audio/mp4; codecs=\\"mp4a.40.3\\"":{"canPlay":"","mediaSource":false},"audio/mp4; codecs=\\"mp4a.40.4\\"":{"canPlay":"","mediaSource":false},"audio/mp4; codecs=\\"mp4a.40.5\\"":{"canPlay":"probably","mediaSource":true},"audio/mp4; codecs=\\"mp4a.40.6\\"":{"canPlay":"","mediaSource":false},"audio/mp4; codecs=\\"mp4a.40.7\\"":{"canPlay":"","mediaSource":false},"audio/mp4; codecs=\\"mp4a.40.8\\"":{"canPlay":"","mediaSource":false},"audio/mp4; codecs=\\"mp4a.40.9\\"":{"canPlay":"","mediaSource":false},"audio/mp4; codecs=\\"mp4a.40.12\\"":{"canPlay":"","mediaSource":false},"audio/mp4; codecs=\\"mp4a.40.13\\"":{"canPlay":"","mediaSource":false},"audio/mp4; codecs=\\"mp4a.40.14\\"":{"canPlay":"","mediaSource":false},"audio/mp4; codecs=\\"mp4a.40.15\\"":{"canPlay":"","mediaSource":false},"audio/mp4; codecs=\\"mp4a.40.16\\"":{"canPlay":"","mediaSource":false},"audio/mp4; codecs=\\"mp4a.40.17\\"":{"canPlay":"","mediaSource":false},"audio/mp4; codecs=\\"mp4a.40.19\\"":{"canPlay":"","mediaSource":false},"audio/mp4; codecs=\\"mp4a.40.20\\"":{"canPlay":"","mediaSource":false},"audio/mp4; codecs=\\"mp4a.40.21\\"":{"canPlay":"","mediaSource":false},"audio/mp4; codecs=\\"mp4a.40.22\\"":{"canPlay":"","mediaSource":false},"audio/mp4; codecs=\\"mp4a.40.23\\"":{"canPlay":"","mediaSource":false},"audio/mp4; codecs=\\"mp4a.40.24\\"":{"canPlay":"","mediaSource":false},"audio/mp4; codecs=\\"mp4a.40.25\\"":{"canPlay":"","mediaSource":false},"audio/mp4; codecs=\\"mp4a.40.26\\"":{"canPlay":"","mediaSource":false},"audio/mp4; codecs=\\"mp4a.40.27\\"":{"canPlay":"","mediaSource":false},"audio/mp4; codecs=\\"mp4a.40.28\\"":{"canPlay":"","mediaSource":false},"audio/mp4; codecs=\\"mp4a.40.29\\"":{"canPlay":"probably","mediaSource":true},"audio/mp4; codecs=\\"mp4a.40.32\\"":{"canPlay":"","mediaSource":false},"audio/mp4; codecs=\\"mp4a.40.33\\"":{"canPlay":"","mediaSource":false},"audio/mp4; codecs=\\"mp4a.40.34\\"":{"canPlay":"","mediaSource":false},"audio/mp4; codecs=\\"mp4a.40.35\\"":{"canPlay":"","mediaSource":false},"audio/mp4; codecs=\\"mp4a.40.36\\"":{"canPlay":"","mediaSource":false},"audio/mp4; codecs=\\"mp4a.66\\"":{"canPlay":"probably","mediaSource":false},"audio/mp4; codecs=\\"mp4a.67\\"":{"canPlay":"probably","mediaSource":true},"audio/mp4; codecs=\\"mp4a.68\\"":{"canPlay":"probably","mediaSource":false},"audio/mp4; codecs=\\"mp4a.69\\"":{"canPlay":"probably","mediaSource":false},"audio/mp4; codecs=\\"mp4a.6B\\"":{"canPlay":"probably","mediaSource":false},"audio/mp4; codecs=\\"mp3\\"":{"canPlay":"probably","mediaSource":false},"audio/mp4; codecs=\\"flac\\"":{"canPlay":"probably","mediaSource":true},"audio/mp4; codecs=\\"bogus\\"":{"canPlay":"","mediaSource":false},"audio/mp4; codecs=\\"aac\\"":{"canPlay":"","mediaSource":false},"audio/mp4; codecs=\\"ac3\\"":{"canPlay":"","mediaSource":false},"audio/mp4; codecs=\\"A52\\"":{"canPlay":"","mediaSource":false},"audio/mpeg; codecs=\\"mp3\\"":{"canPlay":"probably","mediaSource":false},"audio/wav; codecs=\\"0\\"":{"canPlay":"","mediaSource":false},"audio/wav; codecs=\\"2\\"":{"canPlay":"","mediaSource":false},"audio/wave; codecs=\\"0\\"":{"canPlay":"","mediaSource":false},"audio/wave; codecs=\\"1\\"":{"canPlay":"","mediaSource":false},"audio/wave; codecs=\\"2\\"":{"canPlay":"","mediaSource":false},"audio/x-wav; codecs=\\"0\\"":{"canPlay":"","mediaSource":false},"audio/x-wav; codecs=\\"1\\"":{"canPlay":"probably","mediaSource":false},"audio/x-wav; codecs=\\"2\\"":{"canPlay":"","mediaSource":false},"audio/x-pn-wav; codecs=\\"0\\"":{"canPlay":"","mediaSource":false},"audio/x-pn-wav; codecs=\\"1\\"":{"canPlay":"","mediaSource":false},"audio/x-pn-wav; codecs=\\"2\\"":{"canPlay":"","mediaSource":false}}')
            },
            {
                "key": "video_codecs",
                "value": "{\"ogg\":\"\",\"h264\":\"probably\",\"webm\":\"probably\",\"mpeg4v\":\"\",\"mpeg4a\":\"\",\"theora\":\"\"}"
            },
            {
                "key": "video_codecs_extended_hash",
                "value": self.md5_hash(
                    '{"video/mp4; codecs=\\"hev1.1.6.L93.90\\"":{"canPlay":"probably","mediaSource":true},"video/mp4; codecs=\\"hvc1.1.6.L93.90\\"":{"canPlay":"probably","mediaSource":true},"video/mp4; codecs=\\"hev1.1.6.L93.B0\\"":{"canPlay":"probably","mediaSource":true},"video/mp4; codecs=\\"hvc1.1.6.L93.B0\\"":{"canPlay":"probably","mediaSource":true},"video/mp4; codecs=\\"vp09.00.10.08\\"":{"canPlay":"probably","mediaSource":true},"video/mp4; codecs=\\"vp09.00.50.08\\"":{"canPlay":"probably","mediaSource":true},"video/mp4; codecs=\\"vp09.01.20.08.01\\"":{"canPlay":"probably","mediaSource":true},"video/mp4; codecs=\\"vp09.01.20.08.01.01.01.01.00\\"":{"canPlay":"probably","mediaSource":true},"video/mp4; codecs=\\"vp09.02.10.10.01.09.16.09.01\\"":{"canPlay":"probably","mediaSource":true},"video/mp4; codecs=\\"av01.0.08M.08\\"":{"canPlay":"probably","mediaSource":true},"video/webm; codecs=\\"vorbis\\"":{"canPlay":"probably","mediaSource":true},"video/webm; codecs=\\"vp8\\"":{"canPlay":"probably","mediaSource":true},"video/webm; codecs=\\"vp8.0\\"":{"canPlay":"probably","mediaSource":false},"video/webm; codecs=\\"vp8.0, vorbis\\"":{"canPlay":"probably","mediaSource":false},"video/webm; codecs=\\"vp8, opus\\"":{"canPlay":"probably","mediaSource":true},"video/webm; codecs=\\"vp9\\"":{"canPlay":"probably","mediaSource":true},"video/webm; codecs=\\"vp9, vorbis\\"":{"canPlay":"probably","mediaSource":true},"video/webm; codecs=\\"vp9, opus\\"":{"canPlay":"probably","mediaSource":true},"video/x-matroska; codecs=\\"theora\\"":{"canPlay":"","mediaSource":false},"application/x-mpegURL; codecs=\\"avc1.42E01E\\"":{"canPlay":"","mediaSource":false},"video/ogg; codecs=\\"dirac, vorbis\\"":{"canPlay":"","mediaSource":false},"video/ogg; codecs=\\"theora, speex\\"":{"canPlay":"","mediaSource":false},"video/ogg; codecs=\\"theora, vorbis\\"":{"canPlay":"","mediaSource":false},"video/ogg; codecs=\\"theora, flac\\"":{"canPlay":"","mediaSource":false},"video/ogg; codecs=\\"dirac, flac\\"":{"canPlay":"","mediaSource":false},"video/ogg; codecs=\\"flac\\"":{"canPlay":"probably","mediaSource":false},"video/3gpp; codecs=\\"mp4v.20.8, samr\\"":{"canPlay":"","mediaSource":false}}')
            },
            {
                "key": "media_query_dark_mode",
                "value": False
            },
            {
                "key": "css_media_queries",
                "value": 0
            },
            {
                "key": "css_color_gamut",
                "value": "srgb"
            },
            {
                "key": "css_contrast",
                "value": "no-preference"
            },
            {
                "key": "css_monochrome",
                "value": False
            },
            {
                "key": "css_pointer",
                "value": "fine"
            },
            {
                "key": "css_grid_support",
                "value": False
            },
            {
                "key": "headless_browser_phantom",
                "value": False
            },
            {
                "key": "headless_browser_selenium",
                "value": False
            },
            {
                "key": "headless_browser_nightmare_js",
                "value": False
            },
            {
                "key": "headless_browser_generic",
                "value": 4
            },
            {
                "key": "1l2l5234ar2",
                "value": str(int(time_now * 1000)) + "⁣"
            },
            {
                "key": "document__referrer",
                "value": self.siteurl + "/"
            },
            {
                "key": "window__ancestor_origins",
                "value": self.window__ancestor_origins
            },
            {
                "key": "window__tree_index",
                "value": self.window__tree_index
            },
            {
                "key": "window__tree_structure",
                "value": self.window__tree_structure
            },
            {
                "key": "window__location_href",
                "value": f"{self.apiurl}/v2/{self.capi_version}/enforcement.{self.enforcement_hash}.html"
            },
            {
                "key": "client_config__sitedata_location_href",
                "value": self.client_config__sitedata_location_href
            },
            {
                "key": "client_config__language",
                "value": None
            },
            {
                "key": "client_config__surl",
                "value": self.apiurl
            },
            {
                "key": "c8480e29a",
                "value": self.md5_hash(self.apiurl) + "⁢"
            },
            {
                "key": "client_config__triggered_inline",
                "value": False
            },
            {
                "key": "mobile_sdk__is_sdk",
                "value": False
            },
            {
                "key": "audio_fingerprint",
                "value": webgl["audio_fingerprint"]
            },
            {
                "key": "navigator_battery_charging",
                "value": True
            },
            {
                "key": "media_device_kinds",
                "value": ["audioinput", "videoinput", "audiooutput"]
            },
            {
                "key": "media_devices_hash",
                "value": "199eba60310b53c200cc783906883c67"
            },
            {
                "key": "navigator_permissions_hash",
                "value": self.md5_hash(
                    "accelerometer|background-sync|camera|clipboard-read|clipboard-write|geolocation|gyroscope|magnetometer|microphone|midi|notifications|payment-handler|persistent-storage")
            },
            {
                "key": "math_fingerprint",
                "value": self.md5_hash(
                    "1.4474840516030247,0.881373587019543,1.1071487177940904,0.5493061443340548,1.4645918875615231,-0.40677759702517235,-0.6534063185820197,9.199870313877772e+307,1.718281828459045,100.01040630344929,0.4828823513147936,1.9275814160560204e-50,7.888609052210102e+269,1.2246467991473532e-16,-0.7181630308570678,11.548739357257748,9.199870313877772e+307,-3.3537128705376014,0.12238344189440875")
            },
            {
                "key": "supported_math_functions",
                "value": self.md5_hash(
                    "abs,acos,acosh,asin,asinh,atan,atanh,atan2,ceil,cbrt,expm1,clz32,cos,cosh,exp,floor,fround,hypot,imul,log,log1p,log2,log10,max,min,pow,random,round,sign,sin,sinh,sqrt,tan,tanh,trunc")
            },
            {
                "key": "screen_orientation",
                "value": "landscape-primary"
            },
            {
                "key": "rtc_peer_connection",
                "value": 5
            },
            {
                "key": "4b4b269e68",
                "value": str(uuid.uuid4())  # window.location.hash
            },
            {
                "key": "6a62b2a558",
                "value": self.enforcement_hash
            },
            {
                "key": "speech_default_voice",
                "value": "Microsoft David - English (United States) || en-US"
            },
            {
                "key": "speech_voices_hash",
                "value": str(uuid.uuid4().hex)
            },
            {
                "key": "4ca87df3d1",
                "value": "Ow=="
            },
            {
                "key": "867e25e5d4",
                "value": "Ow=="
            },
            {
                "key": "d4a306884c",
                "value": "Ow=="
            }]

        for item in enhanced_fp_more:
            enhanced_fp.append(item)

        fp = [
            {
                "key": "api_type",
                "value": "js"
            },
            {
                "key": "f",
                "value": gctx.call("x64hash128", self.process_fp(fp1), 0)
            },
            {
                "key": "n",
                "value": base64.b64encode(str(int(time_now)).encode()).decode()
            },
            {
                "key": "wh",
                "value": f"{str(uuid.uuid4().hex)}|72627afbfd19a741c7da1732218301ac"
            },
            {
                "key": "enhanced_fp",
                "value": enhanced_fp
            },
            {
                "key": "fe",
                "value": fp1
            },
            {
                "key": "ife_hash",
                "value": gctx.call("x64hash128", ", ".join(fp1), 38)
            },
            {
                "key": "jsbd",
                "value": "{\"HL\":6,\"NCE\":true,\"DT\":\"\",\"NWD\":\"false\",\"DMTO\":1,\"DOTO\":1}"
            }
        ]

        return base64.b64encode(Arkose.make_encrypted_dict(self, json.dumps(fp, separators=(',', ':'),
                                                                            ensure_ascii=False)).encode()).decode()

    def _generate_challenge(self) -> None:
        random.seed(random.randint(0, 2 ** 64 - 1))

        task = {
            "bda": self.bda(),
            "public_key": self.sitekey,
            "site": self.siteurl,
            "userbrowser": self.useragent,
            "capi_version": self.capi_version,
            "capi_mode": "inline",
            "style_theme": "default",
            "rnd": str(random.uniform(0, 1)),
            'language': self.locale.split("-")[0]
        }

        if self.blob:
            task["data[blob]"] = self.blob
        headers = {
            'accept-language': f'en-US,en;q=0.9,{self.locale};q=0.8,{self.locale.split("-")[0]};q=0.7',
            'content-type': 'application/x-www-form-urlencoded; charset=UTF-8',
            'origin': self.apiurl,
            **self.base_headers,
            'x-ark-esync-value': self.x_ark_value,
        }
        return self.session.post(
            f'{self.apiurl}/fc/gt2/public_key/{self.sitekey}',
            data=task,
            headers=headers

        ).json()


solver_stats_instance = solver_stats()
threading.Thread(target=solver_stats_instance.title).start()
threading.Thread(target=solver_stats_instance.calc_cpm).start()

max_workers = 50
executor = ThreadPoolExecutor(max_workers=max_workers)
task_queue = queue.Queue()


class preset:
    def get(self):
        return {
            "outlook_register": {
                "siteurl": "https://iframe.arkoselabs.com",
                "sitekey": "B7D8911C-5CC8-A9A3-35B0-554ACEE604DA",
                "apiurl": "https://client-api.arkoselabs.com",
                "data": {
                    "window__ancestor_origins": [
                        "https://iframe.arkoselabs.com",
                        "https://signup.live.com"
                    ],
                    "client_config__sitedata_location_href": "https://iframe.arkoselabs.com/B7D8911C-5CC8-A9A3-35B0-554ACEE604DA/index.html",
                    "window__tree_structure": "[[],[],[[]]]",
                    'window__tree_index': [2, 0]
                },
            },

        }


class TokenManager:
    def __init__(self):
        self.tokens = {
            "123456": {"balance": 99999999.0},
        }
        self.token_lock = Lock()

    def validate_token(self, token: str) -> Dict[str, float]:
        with self.token_lock:
            if token not in self.tokens:
                raise HTTPException(status_code=401, detail="Invalid API token")
            return self.tokens[token]

    def deduct_balance(self, token: str, amount: float = 1.0) -> None:
        with self.token_lock:
            if self.tokens[token]["balance"] < amount:
                raise HTTPException(status_code=403, detail="Insufficient balance")
            self.tokens[token]["balance"] -= amount

    def get_balance(self, token: str) -> float:
        with self.token_lock:
            return self.tokens[token]["balance"]


class TaskCleaner:
    def __init__(self, task_manager, max_age_hours=24, cleanup_interval_minutes=30):
        self.task_manager = task_manager
        self.max_age_hours = max_age_hours
        self.cleanup_interval = cleanup_interval_minutes * 60  # Convert to seconds
        self.running = False
        self.cleanup_thread = None

    def start(self):
        if not self.running:
            self.running = True
            self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
            self.cleanup_thread.start()

    def stop(self):
        self.running = False
        if self.cleanup_thread:
            self.cleanup_thread.join()

    def _cleanup_loop(self):

        while self.running:
            try:
                self._perform_cleanup()
            except Exception as e:
                logging.error(f"Error in task cleanup: {e}")
            time.sleep(self.cleanup_interval)

    def _perform_cleanup(self):

        current_time = datetime.now()
        tasks_to_remove = []

        with self.task_manager.lock:
            for task_id, task_data in self.task_manager.tasks.items():

                if task_data["status"] == "Working":
                    continue

                if "finished" in task_data:
                    finished_time = datetime.fromisoformat(task_data["finished"].replace("Z", "+00:00"))
                    age = (current_time - finished_time).total_seconds() / 3600  # Convert to hours

                    if age >= self.max_age_hours:
                        tasks_to_remove.append(task_id)

            for task_id in tasks_to_remove:
                del self.task_manager.tasks[task_id]

        if tasks_to_remove:
            logging.info(f"Cleaned up {len(tasks_to_remove)} old tasks")


class TaskManager:
    def __init__(self, max_age_hours=24, cleanup_interval_minutes=30):
        self.tasks = {}
        self.lock = Lock()
        self.cleaner = TaskCleaner(self, max_age_hours, cleanup_interval_minutes)
        self.cleaner.start()

    def add_task(self, task_id, task_data):
        with self.lock:
            self.tasks[task_id] = {
                "data": task_data,
                "status": "Working",
                "response": None,
                "attempts": 0,
                "error": None,
                "created": datetime.utcnow().isoformat() + "Z"
            }

    def update_task(self, task_id: str, status: str, response: Dict = None, error: str = None):
        with self.lock:
            if task_id in self.tasks:
                self.tasks[task_id].update({
                    "status": status,
                    "response": response,
                    "error": error,
                    "finished": datetime.utcnow().isoformat() + "Z"
                })

    def get_task(self, task_id: str) -> Dict:
        with self.lock:
            return self.tasks.get(task_id)

    def get_stats(self) -> Dict:
        with self.lock:
            total_tasks = len(self.tasks)
            working_tasks = sum(1 for task in self.tasks.values() if task["status"] == "Working")
            completed_tasks = sum(1 for task in self.tasks.values() if task["status"] == "Success")
            failed_tasks = sum(1 for task in self.tasks.values() if task["status"] == "Failed")

            return {
                "total_tasks": total_tasks,
                "working_tasks": working_tasks,
                "completed_tasks": completed_tasks,
                "failed_tasks": failed_tasks
            }

    def __del__(self):
        if hasattr(self, 'cleaner'):
            self.cleaner.stop()


app = FastAPI()
token_manager = TokenManager()
task_manager = TaskManager(max_age_hours=2, cleanup_interval_minutes=3)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ConcurrentTaskProcessor:
    def __init__(self, max_workers=50):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.task_queue = queue.Queue()
        self._start_workers()

    def _start_workers(self):
        self.workers = []
        for _ in range(50):
            worker = threading.Thread(target=self._worker_thread)
            worker.daemon = True
            worker.start()
            self.workers.append(worker)

    def _worker_thread(self):
        while True:
            task_id, task_data, api_token = self.task_queue.get()
            try:
                self._process_task(task_id, task_data, api_token)
            except Exception as e:
                logging.error(f"Task processing error: {e}")
            finally:
                self.task_queue.task_done()

    def _process_task(self, task_id: str, task_data: Dict, api_token: str):
        try:
            preset_data = None
            if task_data["siteKey"] == "B7D8911C-5CC8-A9A3-35B0-554ACEE604DA":
                preset_data = preset().get()["outlook_register"]

            if not preset_data:
                raise ValueError(f"Unsupported siteKey: {task_data['siteKey']}")

            solver = Funcaptcha(
                apiurl=preset_data["apiurl"],
                siteurl=preset_data["siteurl"],
                sitekey=task_data["siteKey"],
                data=preset_data["data"],
                blob=task_data.get("data", ""),
                custom_cookies=None,
                custom_locale=None,
                proxy="http://127.0.0.1:7890"
            )

            result = solver.solve()

            if result.get("success"):
                response = {
                    "token": result["token"],
                    "details": {
                        "instruction": result.get("game", ""),
                        "waveCount": result.get("waves", 0)
                    }
                }
                task_manager.update_task(task_id, "Success", response)
                token_manager.deduct_balance(api_token)
            else:
                error_msg = result.get("err", "Failed to solve captcha")
                task_manager.update_task(task_id, "Failed", error=error_msg)

        except Exception as e:
            logging.error(f"Task processing error: {str(e)}")
            task_manager.update_task(task_id, "Failed", error=str(e))

    def add_task(self, task_id: str, task_data: Dict, api_token: str):
        self.task_queue.put((task_id, task_data, api_token))


concurrent_processor = ConcurrentTaskProcessor(max_workers=50)


@app.post("/v2/tasks")
async def create_task(request: Request, authorization: str = Header(...)):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid Authorization header")

    api_token = authorization[7:]
    token_manager.validate_token(api_token)

    try:
        body = await request.json()
        task_id = str(uuid.uuid4())

        if not all(k in body for k in ["captchaType", "siteReferer", "siteKey"]):
            raise HTTPException(status_code=400, detail="Missing required fields")

        if body["captchaType"] != "FunCaptcha":
            raise HTTPException(status_code=400, detail="Invalid captcha type")

        task_data = {
            "taskId": task_id,
            "price": "0.01",
            "deducted": False,
            "captchaType": "FunCaptcha",
            "request": body,
            "status": "Working",
            "created": datetime.utcnow().isoformat() + "Z",
            "timestamp": time.time(),
            "token": api_token,
            "siteReferer": body["siteReferer"],
            "siteKey": body["siteKey"],
            "data": body.get("data")
        }

        task_manager.add_task(task_id, task_data)
        concurrent_processor.add_task(task_id, task_data, api_token)

        task_data["remainingBalance"] = token_manager.get_balance(api_token)
        return JSONResponse(task_data)

    except Exception as e:
        logging.error(f"Create task error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/v2/tasks/{task_id}")
async def get_task(task_id: str, authorization: str = Header(...)):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid Authorization header")

    api_token = authorization[7:]
    token_manager.validate_token(api_token)

    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    response = {
        "status": task["status"],
        "task_id": task_id
    }

    if task["status"] == "Success":
        response["response"] = task["response"]
    elif task["error"]:
        response["error"] = task["error"]

    return JSONResponse(response)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=6699)
