# torch
import torch
from torch.utils.data import DataLoader
# huggingface
from transformers import AutoTokenizer, AutoModelForTokenClassification, AdamW, get_scheduler
from datasets import load_dataset
# miscellaneous
from tqdm import tqdm
import os
# mine
from dataset import PuncDataset

def punctuate_text(text, label, task_type="punc"):
    marks = ["", "/"] if task_type == "punc" else ["", ] + list("，。、？！：；")
    return "".join([c + marks[lab] for c, lab in zip(text, label)])


def demo(model, tokenizer, text, task_type="punc"):
    inputs = tokenizer(text, padding=True, return_tensors='pt')
    inputs = {k:inputs[k].to(device) for k in inputs}
    logits = model(**inputs)['logits']
    preds = logits.argmax(dim=-1)
    output = []
    if type(text) is list:
        for s, pred in zip(text, preds):
            output.append(punctuate_text(s, pred[1:-1].cpu().tolist(), task_type=task_type))
        return output
    else:
        return punctuate_text(text, preds[0, 1:-1].cpu().tolist(), task_type=task_type)


if __name__ == '__main__':
    
    # 训练集：史记 汉书 后汉书
    # 2020 全国一（宋史），全国二（宋史），全国三（晋书）
    # （2019全国卷为啥全都是史记……）
    text = [
        "三年权知礼部贡举会大雪苦寒士坐庭中噤未能言轼宽其禁约使得尽技巡铺内侍每摧辱举子且持暖昧单词诬以为罪轼尽奏逐之",
        "开封逻卒夜迹盗盗脱去民有惊出与卒遇缚以为盗民讼诸府不胜考掠之惨遂诬服安中廉知之按得冤状即出民抵吏罪",
        "答曰中兴以来郊祀往往有赦愚意尝谓非宜何者黎庶不达其意将谓效祀必赦至此时凶愚之辈复生心于侥幸矣遂从之",
        "石室诗士施氏嗜狮誓食十狮施氏时时适市视狮十时适十狮适市是时适施氏适市施氏视是十狮恃矢势使是十狮逝世氏拾是十狮尸适石室石室湿氏使侍拭石室石室拭施氏始试食是十狮尸食时始识是十狮尸实十石狮尸试释是事",
        "季姬寂集鸡鸡即棘鸡棘鸡饥叽季姬及箕稷济鸡鸡既济跻姬笈季姬忌急咭鸡鸡急继圾几季姬急即籍箕击鸡箕疾击几伎伎即齑鸡叽集几基季姬急极屐击鸡鸡既殛季姬激即记季姬击鸡记",
        "余不擅文言文五分之翻译题辄得一分而归然临别之际尤不舍机房乃斗胆戏作此篇贻笑大方机房者五楼计算机教室也冬有暖气之温夏有空调之爽怡怡然勉学之佳所也然其境亦广为邻班学友所知每至午休蜂拥而至或议论作业之题或排练心理英语配音之剧不一而足嘈然若市炎夏尤甚吾等常住人口唯有锁门以御外敌而已机房北有白板一面使用经年墨垢固结非以有机溶剂如乙醇者无可拭净者南有队旗上书代码颇显廿四信竞无所畏惧之精神然不知何年二竖子来学于此以白板笔书其名于旗上濯不可去化竞奇才罗伟梁闻之试之以氧化剂还原剂种种皆未果后遂无问津者余每论此事未尝不叹息痛恨于二子也后教练钢哥仿制一旗悬诸南墙然原图失传久矣仿制图糊然若马赛克余尝与胡颖先生合影于此旗前相片竟见嫌不得用",
        "灭人之国必先去其史隳人之枋败人之纲纪必先去其史绝人之材湮塞人之教必先去其史夷人之祖宗必先去其史",
        "先生年四十七在黄州寓居临皋亭就东坡筑雪堂自号东坡居士以东坡图考之自黄州门南至雪堂四百三十步",
    ]
    '''
    2020 
    全国一（宋史）
        gt:       三年/权知礼部贡举/会大雪苦寒/士坐庭中/噤未能言/轼宽其禁约/使得尽技/巡铺内侍每摧辱举子/且持暖昧单词/诬以为罪/轼尽奏逐之/
        jiayan:   三年/权知礼部贡举/会大雪/苦寒士坐庭中/噤未能言/轼宽其禁约/使得尽技巡铺/内侍每摧辱举子/且持暖昧单词/诬以为罪/轼尽奏逐之/
        shanduan: 三年/权知礼部贡举/会大雪苦寒/士坐庭中/噤未能言/轼宽其禁约/使得尽技/巡铺内侍每摧辱举子/且持暖昧单词/诬以为罪/轼尽奏逐之/
        
    全国二（宋史）
        gt:       开封逻卒夜迹盗/盗脱去/民有惊出与卒遇/缚以为盗/民讼诸府/不胜考掠之惨/遂诬服/安中廉知之/按得冤状/即出民/抵吏罪/
        jiayan:   开封逻卒夜迹盗/盗脱去/民有惊出与卒遇/缚以为盗/民讼诸府/不胜考掠之惨/遂诬服/安中廉知之/按得冤状/即出民/抵吏罪/
        shanduan: 开封逻卒夜迹盗/盗脱去/民有惊出与卒遇/缚以为盗/民讼诸府/不胜考掠之惨/遂诬服/安中廉知之/按得冤状/即出民/抵吏罪/
    全国三（晋书）
        gt:       答曰/中兴以来/郊祀往往有赦/愚意尝谓非宜/何者/黎庶不达其意/将谓郊祀必赦/至此时/凶愚之辈复生心于侥幸矣/遂从之/
        jiayan:   答曰/中兴以来/郊祀往往有赦/愚意尝谓非宜/何者/黎庶不达其意/将谓效祀/必赦至此/时凶愚之辈/复生心于侥幸矣/遂从之/
        shanduan: 答曰/中兴以来/郊祀往往有赦/愚意尝谓非宜/何者/黎庶不达其意/将谓效祀必赦/至此时凶愚之辈复生心于侥幸矣/遂从之/
    
    施氏食狮史
        gt:       石室诗士施氏/嗜狮/誓食十狮/施氏时时适市视狮/十时/适十狮适市/是时/适施氏适市/施氏视是十狮/恃矢势/使是十狮逝世/氏拾是十狮尸/适石室/石室湿/氏使侍拭石室/石室拭/施氏始试食是十狮尸/食时/始识是十狮尸/实十石狮尸/试释是事/
        jiayan:   石室诗士/施氏嗜狮/誓食十狮/施氏时时适市/视狮十时适十狮/适市/是时适施氏/适市施氏/视是十狮/恃矢势使/是十狮逝/世氏拾是十/狮尸适石室/石室湿氏/使侍拭石室/石室拭施氏/始试食是十狮尸/食时始识是十/狮尸实十石/狮尸试释是事/
        shanduan: 石室诗士施氏嗜狮/誓食十狮/施氏时时适市/视狮十/时适十狮适市/是时适施氏适市/施氏视是十狮/恃矢势/使是十狮逝/世氏拾是十狮尸/适石室/石室湿氏使侍拭石室/石室拭施氏/始试食是十狮尸/食时始识是十狮尸/实十石狮尸/试释是事/
    
    季姬击鸡记
        gt:       季姬寂/集鸡/鸡即棘鸡/棘鸡饥叽/季姬及箕稷济鸡/鸡既济/跻姬笈/季姬忌/急咭鸡/鸡急/继圾几/季姬急/即籍箕击鸡/箕疾击几伎/伎即齑/鸡叽集几基/季姬急极屐击鸡/鸡既殛/季姬激/即记/季姬击鸡记/
        jiayan:   季姬寂集/鸡鸡即棘鸡/棘鸡饥/叽季姬及箕/稷济鸡/鸡既济跻姬笈/季姬忌急/咭鸡鸡急继圾/几季姬急/即籍箕击鸡箕/疾击几伎伎/即齑鸡叽集/几基季姬/急极屐击鸡/鸡既殛/季姬激即记/季姬击鸡记
        shanduan: 季姬寂集鸡/鸡即棘鸡/棘鸡饥叽/季姬及箕/稷济鸡/鸡既济/跻姬笈/季姬忌急咭鸡/鸡急继圾几/季姬急/即籍箕击鸡/箕疾击几伎/伎即齑鸡叽集几基/季姬急/极屐击鸡/鸡既殛/季姬激/即记季姬击鸡记
        
    '''

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained("ethanyt/guwenbert-base")
    
    task_type = "punc" # "punc" or "punc_type"
    # model = torch.load('./training_logs/model_3_weight1and5/checkpoints/checkpoint10.pth')
    # model = torch.load('./training_logs/model_6_punc_unweighted/checkpoints/checkpoint50.pth')
    model = torch.load('./training_logs/model_7_punc_lunyu/checkpoints/checkpoint50.pth')
    result = demo(model, tokenizer, text, task_type=task_type)
    print(result)

    task_type = "punc_type" # "punc" or "punc_type"
    # model = torch.load('./training_logs/model_4_punc_type/checkpoints/checkpoint50.pth')
    model = torch.load('./training_logs/model_5_punc_type_unweighted/checkpoints/checkpoint45.pth')
    result = demo(model, tokenizer, text, task_type=task_type)
    print(result)
    