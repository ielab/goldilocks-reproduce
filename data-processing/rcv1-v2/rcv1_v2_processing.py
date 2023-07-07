import os
import xml.dom.minidom
from xml.etree import ElementTree as ET


def getSubDir(path):
    dirname=list()
    for file in os.listdir(path):
        dirname.append(file)
    return dirname


def get_xml_data(path):
    # {"news_id", "title", "text", "code", "date"}
    xmlPath = path
    dom = ET.parse(xmlPath)
    root = dom.getroot()
    news_id = root.attrib['itemid']
    date = root.attrib['date']
    title = root[0].text
    text = ""
    for p in dom.findall('./text/p'):
        text += (p.text)
    code = ""
    for c in dom.findall('./metadata/codes/code'):
        code += (c.attrib['code'] + ",")
    code = code.rstrip(",")
    
    xml_dict = {"news_id": news_id, "title": title, "text": text, "code":code, "date":date}
    
    return xml_dict


def get_xml_csv(path, _dict):
    # {"news_id", "title", "text", "code", "date"}
    xmlPath = path
    dom = ET.parse(xmlPath)
    root = dom.getroot()
    news_id = root.attrib['itemid']
    date = root.attrib['date']
    title = root[0].text
    text = ""
    for p in dom.findall('./text/p'):
        text += (p.text)
    code = ""
    for c in dom.findall('./metadata/codes/code'):
        code += (c.attrib['code'] + ",")
    code = code.rstrip(",")
    
    _dict["news_id"].append(news_id)
    _dict["title"].append(title)
    _dict["text"].append(text)
    _dict["code"].append(code)
    _dict["date"].append(date)