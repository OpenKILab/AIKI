from get_llm import generate_text

import re

# Patterns
SCRIPT_PATTERN = r"<[ ]*script.*?\/[ ]*script[ ]*>"
STYLE_PATTERN = r"<[ ]*style.*?\/[ ]*style[ ]*>"
META_PATTERN = r"<[ ]*meta.*?>"
COMMENT_PATTERN = r"<[ ]*!--.*?--[ ]*>"
LINK_PATTERN = r"<[ ]*link.*?>"
BASE64_IMG_PATTERN = r'<img[^>]+src="data:image/[^;]+;base64,[^"]+"[^>]*>'
SVG_PATTERN = r"(<svg[^>]*>)(.*?)(<\/svg>)"


def replace_svg(html: str, new_content: str = "this is a placeholder") -> str:
    return re.sub(
        SVG_PATTERN,
        lambda match: f"{match.group(1)}{new_content}{match.group(3)}",
        html,
        flags=re.DOTALL,
    )


def replace_base64_images(html: str, new_image_src: str = "#") -> str:
    return re.sub(BASE64_IMG_PATTERN, f'<img src="{new_image_src}"/>', html)


def clean_html(html: str, clean_svg: bool = False, clean_base64: bool = False):
    html = re.sub(
        SCRIPT_PATTERN, "", html, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL
    )
    html = re.sub(
        STYLE_PATTERN, "", html, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL
    )
    html = re.sub(
        META_PATTERN, "", html, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL
    )
    html = re.sub(
        COMMENT_PATTERN, "", html, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL
    )
    html = re.sub(
        LINK_PATTERN, "", html, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL
    )

    if clean_svg:
        html = replace_svg(html)
    if clean_base64:
        html = replace_base64_images(html)
    return html


def create_prompt(
    text: str, tokenizer=None, instruction: str = None, schema: str = None
) -> str:
    """
    Create a prompt for the model with optional instruction and JSON schema.
    """
    if not instruction:
        instruction = "Extract the main content from the given HTML and convert it to Markdown format."
    if schema:
        instruction = "Extract the specified information from a list of news threads and present it in a structured JSON format."
        prompt = f"{instruction}\n```html\n{text}\n```\nThe JSON schema is as follows:```json\n{schema}\n```"
    else:
        prompt = f"{instruction}\n```html\n{text}\n```"

    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

html = "<html><body><h1>Hello, world!</h1></body></html>"

html = """<!DOCTYPE html><!--STATUS OK--><html><head><meta http-equiv="Content-Type" content="text/html;charset=utf-8"><meta http-equiv="X-UA-Compatible" content="IE=edge,chrome=1"><meta content="always" name="referrer"><meta name="theme-color" content="#ffffff"><meta name="description" content="全球领先的中文搜索引擎、致力于让网民更便捷地获取信息，找到所求。百度超过千亿的中文网页数据库，可以瞬间找到相关的搜索结果。"><link rel="shortcut icon" href="https://www.baidu.com/favicon.ico" type="image/x-icon" /><link rel="search" type="application/opensearchdescription+xml" href="/content-search.xml" title="百度搜索" /><link rel="stylesheet" data-for="result" href="https://pss.bdstatic.com/r/www/static/font/cosmic/pc/cos-icon_8bae49a.css"/><link rel="icon" sizes="any" mask href="https://www.baidu.com/favicon.ico"><link rel="dns-prefetch" href="//dss0.bdstatic.com"/><link rel="dns-prefetch" href="//dss1.bdstatic.com"/><link rel="dns-prefetch" href="//ss1.bdstatic.com"/><link rel="dns-prefetch" href="//sp0.baidu.com"/><link rel="dns-prefetch" href="//sp1.baidu.com"/><link rel="dns-prefetch" href="//sp2.baidu.com"/><link rel="dns-prefetch" href="//pss.bdstatic.com"/><link rel="apple-touch-icon-precomposed" href="https://psstatic.cdn.bcebos.com/video/wiseindex/aa6eef91f8b5b1a33b454c401_1660835115000.png"><title>百度一下，你就知道</title><style index="newi" type="text/css">#form .bdsug{top:39px}.bdsug{display:none;position:absolute;width:535px;background:#fff;border:1px solid #ccc!important;_overflow:hidden;box-shadow:1px 1px 3px #ededed;-webkit-box-shadow:1px 1px 3px #ededed;-moz-box-shadow:1px 1px 3px #ededed;-o-box-shadow:1px 1px 3px #ededed}.bdsug li{width:519px;color:#000;font:14px arial;line-height:25px;padding:0 8px;position:relative;cursor:default}.bdsug li.bdsug-s{background:#f0f0f0}.bdsug-store span,.bdsug-store b{color:#7A77C8}.bdsug-store-del{font-size:12px;color:#666;text-decoration:underline;position:absolute;right:8px;top:0;cursor:pointer;display:none}.bdsug-s .bdsug-store-del{display:inline-block}.bdsug-ala{display:inline-block;border-bottom:1px solid #e6e6e6}.bdsug-ala h3{line-height:14px;background:url(//www.baidu.com/img/sug_bd.png?v=09816787.png) no-repeat left center;margin:6px 0 4px;font-size:12px;font-weight:400;color:#7B7B7B;padding-left:20px}.bdsug-ala p{font-size:14px;font-weight:700;padding-left:20px}#m .bdsug .bdsug-direct p{color:#00c;font-weight:700;line-height:34px;padding:0 8px;margin-top:0;cursor:pointer;white-space:nowrap;overflow:hidden}#m .bdsug .bdsug-direct p img{width:16px;height:16px;margin:7px 6px 9px 0;vertical-align:middle}#m .bdsug .bdsug-direct p span{margin-left:8px}#form .bdsug .bdsug-direct{width:auto;padding:0;border-bottom:1px solid #f1f1f1}#form .bdsug .bdsug-direct p i{font-size:12px;line-height:100%;font-style:normal;font-weight:400;color:#fff;background-color:#2b99ff;display:inline;text-align:center;padding:1px 5px;*padding:2px 5px 0;margin-left:8px;overflow:hidden}.bdsug .bdsug-pcDirect{color:#000;font-size:14px;line-height:30px;height:30px;background-color:#f8f8f8}.bdsug .bdsug-pc-direct-tip{position:absolute;right:15px;top:8px;width:55px;height:15px;display:block;background:url(https://pss.bdstatic.com/r/www/cache/static/protocol/https/global/img/pc_direct_42d6311.png) no-repeat 0 0}.bdsug li.bdsug-pcDirect-s{background-color:#f0f0f0}.bdsug .bdsug-pcDirect-is{color:#000;font-size:14px;line-height:22px;background-color:#f5f5f5}.bdsug .bdsug-pc-direct-tip-is{position:absolute;right:15px;top:3px;width:55px;height:15px;display:block;background:url(https://pss.bdstatic.com/r/www/cache/static/protocol/https/global/img/pc_direct_42d6311.png) no-repeat 0 0}.bdsug li.bdsug-pcDirect-is-s{background-color:#f0f0f0}.bdsug .bdsug-pcDirect-s .bdsug-pc-direct-tip,.bdsug .bdsug-pcDirect-is-s .bdsug-pc-direct-tip-is{background-position:0 -15px}.bdsug .bdsug-newicon{color:#929292;opacity:.7;font-size:12px;display:inline-block;line-height:22px;letter-spacing:2px}.bdsug .bdsug-s .bdsug-newicon{opacity:1}.bdsug .bdsug-newicon i{letter-spacing:0;font-style:normal}.bdsug .bdsug-feedback-wrap{display:none}.toggle-underline{text-decoration:none}.toggle-underline:hover{text-decoration:underline}.bdpfmenu,.usermenu{border:1px solid #d1d1d1;position:absolute;width:105px;top:36px;z-index:302;box-shadow:1px 1px 5px #d1d1d1;-webkit-box-shadow:1px 1px 5px #d1d1d1;-moz-box-shadow:1px 1px 5px #d1d1d1;-o-box-shadow:1px 1px 5px #d1d1d1}.bdpfmenu{font-size:12px;background-color:#fff}.bdpfmenu a,.usermenu a{display:block;text-align:left;margin:0!important;padding:0 9px;line-height:26px;text-decoration:none}.briiconsbg{background-repeat:no-repeat;background-size:300px 18px;background-image:url(https://pss.bdstatic.com/r/www/cache/static/protocol/https/home/img/icons_0c37e9b.png);background-image:url(https://pss.bdstatic.com/r/www/cache/static/protocol/https/home/img/icons_809ae65.gif)\9}.bdpfmenu a:link,.bdpfmenu a:visited,#u .usermenu a:link,#u .usermenu a:visited{background:#fff;color:#333}.bdpfmenu a:hover,.bdpfmenu a:active,#u .usermenu a:hover,#u .usermenu a:active{background:#38f;text-decoration:none;color:#fff}.bdpfmenu{width:70px}#wrapper .bdnuarrow{width:0;height:0;font-size:0;line-height:0;display:block;position:absolute;top:-10px;left:50%;margin-left:-5px}#wrapper .bdnuarrow em,#wrapper .bdnuarrow i{width:0;height:0;font-size:0;line-height:0;display:block;position:absolute;border:5px solid transparent;border-style:dashed dashed solid}#wrapper .bdnuarrow em{border-bottom-color:#d8d8d8;top:-1px}#wrapper .bdnuarrow i{border-bottom-color:#fff;top:0}#gxszHead .prefpanelclose{cursor:pointer;width:16px;height:16px;float:right;margin-top:7px;background-position:-248px 0}#gxszHead .prefpanelclose:hover{background-position:-264px 0}.s_ipt::-webkit-input-placeholder{padding-left:3px;color:#aaa;font-size:13px}.s_ipt::-moz-placeholder{padding-left:3px;color:#aaa;font-size:13px}.s_ipt:-ms-input-placeholder{padding-left:3px;color:#aaa;font-size:13px}.s_ipt::placeholder{padding-left:3px;color:#aaa;font-size:13px}.kw-placeholder{position:absolute;top:0;left:0;color:#aaa;font-size:13px;height:40px;line-height:40px;padding-left:10px;max-width:360px;z-index:99;pointer-events:none}.kw-placeholder.kw-placehlder-high{height:40px;line-height:40px}.kw-placeholder.placeholders-hidden{visibility:hidden}#head_wrapper #form .bdsug-new{width:544px;top:35px;border-radius:0 0 10px 10px;border:2px solid #4E6EF2!important;border-top:0!important;box-shadow:none;font-family:Arial,sans-serif;z-index:1}#main-wrapper .bdsug-new{border-radius:12px;position:relative;padding:10px;box-sizing:border-box;border:1px solid #dbdce0!important;box-shadow:0 2px 12px 0 rgba(51,91,255,.08);font-family:Arial,sans-serif;z-index:1;opacity:0;transform:translateY(-9px);transition:transform 160ms ease,opacity 160ms ease}.new_input_superman #main-wrapper .bdsug-new{width:100%}#main-wrapper .bdsug-new .bdsug-feedback-wrap{margin-top:0;height:unset;line-height:16px;margin-bottom:4px;margin-top:8px}#main-wrapper .bdsug-new .bdsug-feedback-wrap span{color:#626675;font-size:14px;margin-right:6px}#main-wrapper .bdsug-new .bdsug-feedback-wrap span:hover{color:#335BFF}#main-wrapper .bdsug-new ul{margin:unset;padding:unset;border-top:unset;background:unset}#main-wrapper .bdsug-new ul li{line-height:34px;font-size:16px;margin-left:0;margin-right:0;padding-left:5px;padding-right:0;width:100%;border-radius:9px;box-sizing:border-box;color:#9195A3}#main-wrapper .bdsug-new ul li .bdsug-store-del{font-size:14px;color:#626675;text-decoration:none;font-weight:400}#main-wrapper .bdsug-new ul li .bdsug-store-del:hover{color:#335BFF}#main-wrapper .bdsug-new ul span{color:#333}#main-wrapper .bdsug-new ul li b{font-weight:400;color:#333;margin-left:3px}#main-wrapper .bdsug-new ul li b.total-b{margin-left:0}#main-wrapper .bdsug-new ul li:hover{cursor:pointer;font-weight:500}#main-wrapper .bdsug-new ul li:hover b,#main-wrapper .bdsug-new ul li:hover span:not(.sug-tag-text){color:#335BFF;font-weight:500}#main-wrapper .bdsug-new ul .bdsug-s,#main-wrapper .bdsug-new ul .bdsug-s b,#main-wrapper .bdsug-new ul .bdsug-s span:not(.sug-tag-text){color:#335BFF;background-color:#F5F6F9!important;font-weight:500}#main-wrapper .bdsug-new .bdsug-feedback-tool-zone{line-height:16px!important;height:unset!important;padding-bottom:4px!important;margin-top:9px!important}#main-wrapper .bdsug-new .bdsug-feedback-tool-zone span{margin-right:15px!important;color:#626675!important}#main-wrapper .bdsug-new .bdsug-feedback-tool-zone a{color:#626675!important}#main-wrapper .bdsug-new .bdsug-feedback-tool-zone span:hover,#main-wrapper .bdsug-new .bdsug-feedback-tool-zone a:hover{color:#335BFF!important}#main-wrapper .bdsug-new .bdsug-feedback-tool-zone>:last-child{margin-right:6px!important}#main-wrapper .bdsug-new ul li .direct-sug-wrap .right-btn{margin-top:7px}#main-wrapper .bdsug-new.enter{opacity:1;transform:translateY(9px)}#main-wrapper .bdsug-new.exit{opacity:0;transform:translateY(9px)}#head_wrapper.sam_head_wrapper2 #form .bdsug-new{width:545px;z-index:1;border:1px solid #4E6EF2!important;border-top:0!important}#head_wrapper #form .bdsug-new ul{margin:7px 14px 0;padding:8px 0 7px;background:0 0;border-top:2px solid #f5f5f6}#head_wrapper #form .bdsug-new ul li{position:relative;width:auto;padding-left:14px;margin-left:-14px;margin-right:-14px;color:#626675;line-height:28px;background:0 0;font-family:Arial,sans-serif}#head_wrapper #form .bdsug-new ul li .sug-search-icon,#head_wrapper #form .bdsug-new ul li .sug-history-icon{margin-right:4px;color:#222}#head_wrapper #form .bdsug-new ul li span{color:#626675}#head_wrapper #form .bdsug-new ul li b{font-weight:400;color:#222}#head_wrapper #form .bdsug-new .bdsug-store-del{font-size:13px;text-decoration:none;color:#9195A3;right:16px}#head_wrapper #form .bdsug-new .bdsug-store-del:hover{color:#315EFB;cursor:pointer}#head_wrapper #form .bdsug-new ul li:hover,#head_wrapper #form .bdsug-new ul li:hover span,#head_wrapper #form .bdsug-new ul li:hover b{cursor:pointer}.wrapper_new #form .bdsug-new .bdsug-s{background-color:#F5F5F6!important}.wrapper_new #form .sam_search .bdsug-new .bdsug-s{background-color:#F1F3FD!important}#head_wrapper #form .sam_search .bdsug-new .bdsug-s{background-color:#F1F3FD!important}#head .s-down #form .bdsug-new{top:32px}.s-skin-hasbg #head_wrapper #form .bdsug-new{border-color:#4569ff!important;border-top:0!important}.s-skin-hasbg #head_wrapper.s-down #form .bdsug-new{border-color:#4e6ef2!important;border-top:0!important}.s-skin-hasbg #head_wrapper.s-down #form.sam_search .bdsug-new{border-color:rgba(0,0,0,.05)!important;border-top:1px solid rgba(0,0,0,.05)!important;top:54px!important}#head_wrapper #form .bdsug-new .bdsug-s,#head_wrapper #form .bdsug-new .bdsug-s span,#head_wrapper #form .bdsug-new .bdsug-s b{color:#315EFB}#head_wrapper #form .bdsug-new>div span:hover,#head_wrapper #form .bdsug-new>div a:hover{color:#315EFB!important}#head_wrapper #form #kw.new-ipt-focus{border-color:#4e6ef2}#head_wrapper #form .bdsug-new ul li,.new_input_superman #main-wrapper .bdsug-new ul li{}#head_wrapper #form .bdsug-new ul li .sug-hot-orange,.new_input_superman #main-wrapper .bdsug-new ul li .sug-hot-orange,#head_wrapper #form .bdsug-new ul li .sug-hot-grey,.new_input_superman #main-wrapper .bdsug-new ul li .sug-hot-grey,#head_wrapper #form .bdsug-new ul li .sug-hot-blue,.new_input_superman #main-wrapper .bdsug-new ul li .sug-hot-blue{display:inline-block;width:12px;height:12px;font-size:12px;line-height:12px;padding:2px;text-align:center;font-weight:500;margin-left:6px;vertical-align:text-bottom}#head_wrapper #form .bdsug-new ul li .sug-hot-orange,.new_input_superman #main-wrapper .bdsug-new ul li .sug-hot-orange{display:inline-block;color:#fff;background:#F60;border-radius:4px}#head_wrapper #form .bdsug-new ul li .sug-new-tag,.new_input_superman #main-wrapper .bdsug-new ul li .sug-new-tag{text-align:center;margin-left:6px;box-sizing:border-box;font-size:12px;line-height:14px;padding:1px 4px;font-weight:500}#head_wrapper #form .bdsug-new ul li .sug-new-tag-grey,.new_input_superman #main-wrapper .bdsug-new ul li .sug-new-tag-grey{color:#858585;border:1px solid rgba(133,133,133,.5);border-radius:4px}#head_wrapper #form .bdsug-new ul li .sug-new-tag-blue,.new_input_superman #main-wrapper .bdsug-new ul li .sug-new-tag-blue{color:#36F;border:1px solid rgba(51,102,255,.4);border-radius:4px}#head_wrapper #form .bdsug-new ul li .sug-new-tag-orange,.new_input_superman #main-wrapper .bdsug-new ul li .sug-new-tag-orange{color:#F33;border:1px solid rgba(255,51,51,.4);border-radius:4px}#head_wrapper #form .bdsug-new ul li .sug-tag-text,.new_input_superman #main-wrapper .bdsug-new ul li .sug-tag-text{position:sticky;right:-4px;margin-top:-2.5px;display:inline-block;box-sizing:border-box;margin-left:6px;padding:1px;font-size:12px;line-height:12px;font-weight:500;text-align:center;border-radius:4px;vertical-align:middle;border-width:1px;border-style:solid}#head_wrapper #form .bdsug-new ul li .sug-tag-img,.new_input_superman #main-wrapper .bdsug-new ul li .sug-tag-img{height:18px;max-width:80px;vertical-align:middle;margin-left:6px}#head_wrapper .bdsug-new ul li .direct-sug-wrap{display:block;height:40px;padding:7px 0;color:#222}#head_wrapper .bdsug-new ul li .direct-sug-wrap:visited{color:#222}#head_wrapper .bdsug-new ul li .direct-sug-wrap .left-img-wrap{position:relative}#head_wrapper .bdsug-new ul li .direct-sug-wrap .left-img-wrap::before{content:"";width:38px;height:38px;position:absolute;top:0;right:0;left:0;bottom:0;border:1px solid rgba(0,0,0,.06);border-radius:9px}#head_wrapper .bdsug-new ul li .direct-sug-wrap .left-img{box-sizing:border-box;float:left;width:40px;height:40px;border:1px solid rgba(0,0,0,.06);border-radius:9px;margin-right:6px}#head_wrapper .bdsug-new ul li .direct-sug-wrap .content{float:left}#head_wrapper .bdsug-new ul li .direct-sug-wrap .header{margin-bottom:6px;line-height:18px;height:18px;vertical-align:middle}#head_wrapper .bdsug-new ul li .direct-sug-wrap .name{float:left;font-size:16px;color:#222;font-weight:500}#head_wrapper .bdsug-new ul li .direct-sug-wrap .tag-img{width:18px;margin-left:4px}#head_wrapper .bdsug-new ul li .direct-sug-wrap .more{line-height:14px;width:240px}#head_wrapper .bdsug-new ul li .direct-sug-wrap .brief,#head_wrapper .bdsug-new ul li .direct-sug-wrap .info{font-size:14px;color:#222;font-weight:400}#head_wrapper .bdsug-new ul li .direct-sug-wrap .brief{margin-right:6px}#head_wrapper .bdsug-new ul li .direct-sug-wrap .right-btn{float:right;margin-top:5px;margin-right:6px;width:60px;height:24px;text-align:center;font-size:14px;color:#36F;line-height:24px;background-image:linear-gradient(111deg,#e8f7ff 0,#edf0ff 100%);border-radius:12px}#head_wrapper .bdsug-new ul li .direct-sug-wrap .right-btn:hover{background:#315efb;color:#fff}#head_wrapper #form .bdsug-new ul li:hover .direct-sug-wrap .name{color:#315EFB}#head_wrapper #form .bdsug-new ul li:hover .direct-sug-wrap .brief{color:#222}#head_wrapper.s-down #form .sam-bdsug.bdsug-new{top:52px}#head_wrapper #form .sam-bdsug.bdsug-new{width:100%;box-shadow:0 4px 4px 0 rgba(0,0,0,.1);border:1px solid rgba(0,0,0,.05)!important;border-radius:12px;top:56px}#head_wrapper #form .sam-bdsug.bdsug-new ul{border:0;padding:0 0 7px}#head_wrapper #form .sam-bdsug.bdsug-new ul li{line-height:32px}#head_wrapper #form .sam-bdsug.bdsug-new ul .bdsug-s{background-color:#F1F3FD!important}#head_wrapper #form .sam-bdsug.bdsug-new .bdsug-store-del{right:15px}.sam_search .sam_search_rec,.sam_search .sam_search_soutu{z-index:1;display:none;position:absolute;top:50%;margin-top:-12px;font-size:24px;color:#4E6EF2;height:24px;line-height:24px;width:24px;cursor:pointer;-webkit-transform:translate3d(0,0,0);transform:translate3d(0,0,0);transition:transform .3s ease}.sam_search .sam_search_rec{right:54px}.sam_search .sam_search_soutu{right:14px}.sam_search .sam_search_rec:hover,.sam_search .sam_search_soutu:hover{color:#1D4FFF!important;transform:scale(1.08,1.08)}.sam_search .sam_search_rec_hover,.sam_search .sam_search_soutu_hover{background:#626675;border-radius:8px;height:32px;width:76px;text-align:center;line-height:32px;font-size:13px;color:#FFF;position:absolute;z-index:2;top:50px}.sam_search .sam_search_rec_hover:before,.sam_search .sam_search_soutu_hover:before{content:'';border:4px solid transparent;border-bottom:4px solid #626675;position:absolute;left:50%;top:-8px;margin-left:-4px}.sam_search .sam_search_rec_hover{right:29px}.sam_search .sam_search_soutu_hover{display:none;right:-12px}</style><style type="text/css" index="head">blockquote,body,button,dd,dl,dt,fieldset,form,h1,h2,h3,h4,h5,h6,hr,input,legend,li,ol,p,pre,td,textarea,th,ul{margin:0;padding:0}
html{color:#000;overflow-y:scroll;overflow:-moz-scrollbars}
body,button,input,select,textarea{font-size:12px;font-family:Arial,sans-serif}
h1,h2,h3,h4,h5,h6{font-size:100%}
em{font-style:normal}
small{font-size:12px}
ol,ul{list-style:none}
a{text-decoration:none}
a:hover{text-decoration:underline}
legend{color:#000}
fieldset,img{border:0}
button,input,select,textarea{font-size:100%}
table{border-collapse:collapse;border-spacing:0}
img{-ms-interpolation-mode:bicubic}
textarea{resize:vertical}
.left{float:left}
.right{float:right}
.overflow{overflow:hidden}
.hide{display:none}
.block{display:block}
.inline{display:inline}
.error{color:red;font-size:12px}
button,label{cursor:pointer}
.clearfix:after{content:'\20';display:block;height:0;clear:both}
.clearfix{zoom:1}
.clear{clear:both;height:0;line-height:0;font-size:0;visibility:hidden;overflow:hidden}
.wordwrap{word-break:break-all;word-wrap:break-word}
.s-yahei{font-family:arial,'Microsoft Yahei','微软雅黑'}
pre.wordwrap{white-space:pre-wrap}
body{text-align:center;background:#fff;width:100%}
body,form{position:relative;z-index:0}
td{text-align:left}
img{border:0}
#s_wrap{position:relative;z-index:0;min-width:1000px}
#wrapper{height:100%}
#head .s-ps-islite{_padding-bottom:370px}
#head_wrapper.s-ps-islite{padding-bottom:370px}
.new_input_superman .s-upgrade-down .s_form_login.s_form{min-width:1250px;transform:translateX(-132px)}
.new_input_superman .s-upgrade-down .s_form_login.s_form .s_form_wrapper{display:flex;margin:0 auto}
.new_input_superman .s-upgrade-down .s_form_login.s_form .s_form_wrapper #result_logo{margin-top:2px;margin-right:16px}
.new_input_superman .s-upgrade-down .s_form_login.s_form .s_form_wrapper #main-wrapper{animation:moveMargin 160ms ease forwards}
@keyframes moveMargin{
from{transform:translateX(120px)}
to{transform:translateX(0)}
}
#head_wrapper.s-ps-islite #s_lm_wrap{bottom:298px;background:0 0!important;filter:none!important}
#head_wrapper.s-ps-islite .s_form{position:relative;z-index:1}
#head_wrapper.s-ps-islite .fm{position:absolute;bottom:0}
#head_wrapper.s-ps-islite .s-p-top{position:absolute;bottom:40px;width:100%;height:181px}
#head_wrapper.s-ps-islite #s_lg_img,#head_wrapper.s-ps-islite #s_lg_img_aging,#head_wrapper.s-ps-islite #s_lg_img_new{position:static;margin:33px auto 0 auto}
.s_lm_hide{display:none!important}
#head_wrapper.s-down #s_lm_wrap{display:none}
.s-lite-version #m{padding-top:125px}
#s_lg_img,#s_lg_img_aging,#s_lg_img_new{position:absolute;bottom:10px;left:50%;margin-left:-135px}
#form{z-index:1}
.new_input_superman .virtual-form{display:none}
#s_lm_wrap{position:absolute;margin-left:-447px;bottom:0;left:50%;z-index:0;height:30px;width:895px;line-height:30px;text-align:center}
.s-skin-hasbg #s_lm_wrap{background:0 0;background-image:-webkit-gradient(linear,left top,left bottom,from(rgba(0,0,0,.3)),to(rgba(0,0,0,.3)));background-image:-moz-linear-gradient(rgba(0,0,0,.3) 0,rgba(0,0,0,.3) 100%);background-image:-ms-linear-gradient(rgba(0,0,0,.3) 0,rgba(0,0,0,.3) 100%);background-image:-o-linear-gradient(rgba(0,0,0,.3) 0,rgba(0,0,0,.3) 100%);background-image:linear-gradient(rgba(0,0,0,.3) 0,rgba(0,0,0,.3) 100%);filter:progid:DXImageTransform.Microsoft.gradient(startColorstr=#66000000, endColorstr=#66000000)}
#s_lm_wrap.s-down{display:none}
#lm{color:#666;height:15px;line-height:16px;padding:7px 0}
#lm a{text-decoration:underline;color:#666}
#nv{margin:0 0 5px;_margin-bottom:4px;padding:2px 0 0;text-align:left;text-indent:50px}
#nv a,#nv b{margin-left:19px}
#lk,#nv a,#nv b,.btn{font-size:14px}
.s-down .s_form{padding-left:0;margin-top:0;min-height:0}
.s_form .tools{position:absolute;right:-55px}
.s_form_wrapper{height:100%}
#head_wrapper.s-down #mCon span{color:#000}
#lk{margin:33px 0}
#lk span{font:14px "\5b8b\4f53"}
#lh{margin:16px 0 5px;word-spacing:3px}
#mCon{height:15px;line-height:15px;width:28px;padding:10px 8px 0 0;cursor:pointer;background:url('https://pss.bdstatic.com/static/superman/img/spis7-d578e7ff4b.png') no-repeat -684px -605px}
#mCon span{color:#333;cursor:default;display:block}
#mCon .hw{text-decoration:underline;cursor:pointer}
#mMenu{width:56px;border:1px solid #9b9b9b;list-style:none;position:absolute;right:-9px;top:30px;display:none;background:#fff;box-shadow:1px 1px 2px #ccc;-moz-box-shadow:1px 1px 2px #ccc;-webkit-box-shadow:1px 1px 2px #ccc;filter:progid:DXImageTransform.Microsoft.Shadow(Strength=2, Direction=135, Color="#cccccc")\9}
#mMenu a,#mMenu a:visited{color:#00c;width:100%;height:100%;display:block;line-height:22px;text-indent:6px;text-decoration:none;filter:none\9}
#mMenu a:hover{background:#ebebeb}
#mMenu .ln{height:1px;background:#ebebeb;overflow:hidden;font-size:1px;line-height:1px;margin-top:-1px}
#cp,#cp a{color:#77c}
#tb_mr{color:#00c;cursor:pointer;position:relative;z-index:200}
#tb_mr b{font-weight:400}
#nv a,#tb_mr b{text-decoration:underline}
#nv a{color:#00c}
#hwr_div,#loading{z-index:3000}
.bd_bear_home{display:none}
#mHolder{display:none}
#mHolder .c-icon{right:0;top:0;position:absolute;float:right;width:15px;height:15px}
.main{display:none}
#s_feed{display:none}
.s-ps-sug{border:1px solid #ccc!important;box-shadow:1px 1px 3px #ededed;-webkit-box-shadow:1px 1px 3px #ededed;-moz-box-shadow:1px 1px 3px #ededed;-o-box-shadow:1px 1px 3px #ededed;position:absolute;top:32px;left:0}
.s-ps-sug table{width:100%;background:#fff;cursor:default}
.s-ps-sug td{color:#000;font:14px arial;height:25px;line-height:25px;padding:0 8px}
.s-ps-sug td b{color:#000}
.s-ps-sug .mo{background:#ebebeb;cursor:pointer}
.s-ps-sug .ml{background:#fff}
.s-ps-sug td.sug_storage{color:#7a77c8}
.s-ps-sug td.sug_storage b{color:#7a77c8}
.s-ps-sug .sug_del{font-size:12px;color:#666;text-decoration:underline;float:right;cursor:pointer;display:none}
.s-ps-sug .sug_del{font-size:12px;color:#666;text-decoration:underline;float:right;cursor:pointer;display:none}
.s-ps-sug .mo .sug_del{display:block}
.s-ps-sug .sug_ala{border-bottom:1px solid #e6e6e6}
.s-ps-sug td h3{line-height:14px;margin:6px 0 4px 0;font-size:12px;font-weight:400;color:#7b7b7b;padding-left:20px;background:url(img/sug_bd.png) no-repeat left center}
.s-ps-sug td p{font-size:14px;font-weight:700;padding-left:20px}
.s-ps-sug td p span{font-size:12px;font-weight:400;color:#7b7b7b}
#s_user_center{font-weight:400;background-position:right -223px\9}
#s_user_center_menu{right:131px}
.s-ps-islite #nv{padding-top:22px;line-height:16px;height:16px;margin-bottom:13px}
#form .bdsug .bdsug-direct{width:auto;padding:0;border-bottom:1px solid #f1f1f1}
#head_wrapper .bdsug .bdsug-direct p{color:#00c;font-weight:700;line-height:34px;padding:0 8px;margin-top:0;cursor:pointer;white-space:nowrap;overflow:hidden}
#head_wrapper .bdsug .bdsug-direct p img{width:16px;height:16px;margin:7px 6px 9px 0;vertical-align:middle}
#head_wrapper .bdsug .bdsug-direct p span{margin-left:8px}
#head_wrapper .bdsug .bdsug-direct p i{font-size:12px;line-height:100%;font-style:normal;font-weight:400;color:#fff;background-color:#2b99ff;display:inline;text-align:center;padding:1px 5px;*padding:2px 5px 0 5px;margin-left:8px;overflow:hidden}
#result_logo,#s_tab,#u,#wrapper_wrapper{display:none}
#prefpanel{background:#fafafa;display:none;opacity:0;position:fixed;_position:absolute;top:-359px;z-index:500;width:100%;min-width:960px;border-bottom:1px solid #ebebeb;*left:0!important;text-align:left}
#prefpanel form{_width:850px}
@font-face{font-family:cIconfont;src:url('https://pss.bdstatic.com/static/superman/font/iconfont-cdfecb8456.eot');src:url('https://pss.bdstatic.com/static/superman/font/iconfont-cdfecb8456.eot?#iefix') format('embedded-opentype'),url('https://pss.bdstatic.com/static/superman/font/iconfont-fa013548a9.woff2') format('woff2'),url('https://pss.bdstatic.com/static/superman/font/iconfont-840387fb42.woff') format('woff'),url('https://pss.bdstatic.com/static/superman/font/iconfont-4530e108b6.ttf') format('truetype'),url('https://pss.bdstatic.com/static/superman/font/iconfont-74fcdd51ab.svg#iconfont') format('svg')}
.ai-wrapper{width:560px;margin-bottom:40px}
.header-ai-main{display:block;position:relative;margin:0 auto;border-radius:16px;box-sizing:content-box;background:#fff;padding:14px 32px 40px}
.s-news-no-logged-rank-wrapper{top:40px;margin-top:-4px}
.s-news-logged-rank-wrapper{top:26px;z-index:9;margin-top:-4px}
.s-news-logged-rank-wrapper .hot-news-wrapper .s-news-rank-content .title-content .title-content-mark{top:-2px}
.s-news-rank-wrapper{position:absolute;right:0}
.c-gap-top-small{margin-top:3px}
.c-gap-top{margin-top:7px}
.c-gap-top-large{margin-top:11px}
.c-gap-top-mini{margin-top:2px}
.c-gap-top-xsmall{margin-top:4px}
.c-gap-top-middle{margin-top:10px}
.c-gap-bottom-small{margin-bottom:3px}
.c-gap-bottom{margin-bottom:7px}
.c-gap-bottom-large{margin-bottom:11px}
.c-gap-bottom-mini{margin-bottom:2px}
.c-gap-bottom-xsmall{margin-bottom:4px}
.c-gap-bottom-middle{margin-bottom:10px}
.c-gap-left{margin-left:12px}
.c-gap-left-small{margin-left:8px}
.c-gap-left-xsmall{margin-left:4px}
.c-gap-left-mini{margin-left:2px}
.c-gap-left-large{margin-left:16px}
.c-gap-left-middle{margin-left:10px}
.c-gap-right{margin-right:12px}
.c-gap-right-small{margin-right:8px}
.c-gap-right-xsmall{margin-right:4px}
.c-gap-right-mini{margin-right:2px}
.c-gap-right-large{margin-right:16px}
.c-gap-right-middle{margin-right:10px}
.c-gap-icon-right-small{margin-right:5px}
.c-gap-icon-right{margin-right:10px}
.c-gap-icon-left-small{margin-left:5px}
.c-gap-icon-left{margin-left:10px}
.c-row{*zoom:1}
.c-row:after{display:block;height:0;content:"";clear:both;visibility:hidden}
.c-span1{width:32px}
.c-span2{width:80px}
.c-span3{width:128px}
.c-span4{width:176px}
.c-span5{width:224px}
.c-span6{width:272px}
.c-span7{width:320px}
.c-span8{width:368px}
.c-span9{width:416px}
.c-span10{width:464px}
.c-span11{width:512px}
.c-span12{width:560px}
.c-span10,.c-span11,.c-span12,.c-span2,.c-span3,.c-span4,.c-span5,.c-span6,.c-span7,.c-span8,.c-span9{float:left;_display:inline;margin-right:16px;list-style:none}
.c-span-last{margin-right:0}
.c-span-last-s{margin-right:0}
.c-feed-box .c-span1{width:43px}
.c-feed-box .c-span2{width:90px}
.c-feed-box .c-span3{width:137px}
.c-feed-box .c-span4{width:184px}
.c-feed-box .c-span5{width:231px}
.c-feed-box .c-span6{width:278px}
.c-feed-box .c-span7{width:325px}
.c-feed-box .c-span8{width:372px}
.c-feed-box .c-span9{width:419px}
.c-feed-box .c-span10{width:466px}
.c-feed-box .c-span11{width:513px}
.c-feed-box .c-span12{width:560px}
.c-feed-box .c-span10,.c-feed-box .c-span11,.c-feed-box .c-span12,.c-feed-box .c-span2,.c-feed-box .c-span3,.c-feed-box .c-span4,.c-feed-box .c-span5,.c-feed-box .c-span6,.c-feed-box .c-span7,.c-feed-box .c-span8,.c-feed-box .c-span9{margin-right:4px}
.c-feed-box .c-span-last{margin-right:0}
.c-index{display:inline-block;width:14px;padding:1px 0;line-height:100%;text-align:center;color:#fff;background-color:#8eb9f5;font-size:12px}
.c-index-hot,.c-index-hot1{background-color:#f54545}
.c-index-hot2{background-color:#ff8547}
.c-index-hot3{background-color:#ffac38}
.c-index-single{display:inline-block;background:0 0;color:#9195a3;width:18px;font-size:15px;letter-spacing:-1px}
.c-index-single-hot,.c-index-single-hot1{color:#fe2d46}
.c-index-single-hot2{color:#f60}
.c-index-single-hot3{color:#faa90e}
.c-font-sigma{font:36px/60px Arial,sans-serif}
.c-font-large{font:20px/30px Arial,sans-serif}
.c-font-big{font:20px/30px Arial,sans-serif}
.c-font-special{font:16px/26px Arial,sans-serif}
.c-font-medium{font:14px/24px Arial,sans-serif}
.c-font-middle{font:14px/24px Arial,sans-serif}
.c-font-normal{font:13px/23px Arial,sans-serif}
.c-font-small{font:12px/20px Arial,sans-serif}
.c-font-family{font-family:Arial,sans-serif}
.c-color-t{color:#222}
.c-color-text{color:#333}
.c-color-gray{color:#626675}
.c-color-gray2{color:#9195a3}
.c-color-visited{color:#626675}
.c-color-orange{color:#fa4901}
.c-color-green{color:#0ebe90}
.c-color-ad{color:#77a9f9}
.c-color-red{color:#f63051}
.c-color-red:visited{color:#f63051}
.c-color-warn{color:#ff7900}
.c-color-warn:visited{color:#ff7900}
.c-color-link{color:#3951b3}
.c-btn,.c-btn:visited{color:#333!important}
.c-btn{display:inline-block;overflow:hidden;font-family:inherit;font-weight:400;text-align:center;vertical-align:middle;outline:0;border:0;height:30px;width:80px;line-height:30px;font-size:13px;border-radius:6px;padding:0;background-color:#f5f5f6;*zoom:1;cursor:pointer}
.c-btn:hover{background-color:#315efb;color:#fff!important}
a.c-btn{text-decoration:none}
button.c-btn{*overflow:visible;border:0}
button.c-btn::-moz-focus-inner{padding:0;border:0}
.c-btn-disable{color:#c4c7ce!important}
.c-btn-disable:visited{color:#c4c7ce!important}
.c-btn-disable:hover{cursor:default;color:#c4c7ce!important;background-color:#f5f5f6}
.c-btn-mini{height:24px;width:48px;line-height:24px}
.c-btn-mini .c-icon{margin-top:2px}
.c-btn-large{height:30px;line-height:30px;font-size:14px}
button.c-btn-large{height:30px;_line-height:24px}
.c-btn-large .c-icon{margin-top:7px;_margin-top:6px}
.c-btn-primary,.c-btn-primary:visited{color:#fff!important}
.c-btn-primary{background-color:#4e6ef2}
.c-btn-primary:hover{background-color:#315efb}
.new_input_superman .c-btn-primary{background-color:#335bff}
.new_input_superman .c-btn-primary:hover{background-color:#2949cc}
.c-btn-weak{height:24px;line-height:24px;border-radius:4px;font-size:12px}
.c-btn-add{width:32px;height:32px;line-height:32px;text-align:center;color:#9195a3!important}
.c-btn-add:hover{background-color:#4e6ef2;color:#fff!important}
.c-btn-add .c-icon{float:none}
.c-btn-add-disable:hover{cursor:default;color:#c4c7ce!important;background-color:#f5f5f6}
.c-select{position:relative;display:inline-block;width:96px;box-sizing:border-box;-webkit-box-sizing:border-box;-moz-box-sizing:border-box;vertical-align:middle;color:#222;font:13px/23px Arial,sans-serif}
.c-select-selection{display:block;height:30px;line-height:29px;box-sizing:border-box;-webkit-box-sizing:border-box;-moz-box-sizing:border-box;padding:0 26px 0 10px;background-color:#fff;border-radius:6px;border:1px solid #d7d9e0;outline:0;user-select:none;cursor:pointer;position:relative;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.c-select-arrow,.c-select-arrow-up{position:absolute;top:-1px;right:10px;color:#9195a3;font-size:16px}
.c-select-dropdown{display:none;position:absolute;padding-top:4px;top:25px;z-index:999;left:0;width:94px;box-sizing:content-box;-webkit-box-sizing:content-box;-moz-box-sizing:content-box;background:#fff;border-radius:0 0 6px 6px;border:1px solid #d7d9e0;border-top:0;zoom:1}
.c-select-split{border-top:1px solid #f5f5f5;margin:0 5px}
.c-select-dropdown-list{padding:0;margin:5px 0 0;list-style:none}
.c-select-dropdown-list.c-select-scroll{max-height:207px;overflow-y:auto;overflow-x:hidden;margin-right:5px;margin-bottom:9px}
.c-select-dropdown-list.c-select-scroll::-webkit-scrollbar{width:2px}
.c-select-dropdown-list.c-select-scroll::-webkit-scrollbar-track{width:2px;background:#f5f5f6;border-radius:1px}
.c-select-dropdown-list.c-select-scroll::-webkit-scrollbar-thumb{width:2px;height:58px;background-color:#4e71f2;border-radius:1px}
.c-select-dropdown-list.c-select-scroll .c-select-item:last-child{margin:0}
.c-select-item{margin:0 0 4px;padding:0 10px;clear:both;white-space:nowrap;list-style:none;cursor:pointer;box-sizing:border-box;-webkit-box-sizing:border-box;-moz-box-sizing:border-box}
.c-select-item:hover{color:#315efb}
.c-select-item-selected{color:#315efb}
.c-select-arrow-up{display:none}
.c-select-visible .c-select-selection{border-radius:6px 6px 0 0}
.c-select-visible .c-select-dropdown{display:block}
.c-select-visible .c-select-arrow{display:none}
.c-select-visible .c-select-arrow-up{display:inline-block}
.c-img{position:relative;display:block;min-height:1px;border:0;line-height:0;background:#f5f5f6;overflow:hidden}
.c-img img{width:100%}
.c-img1{width:32px}
.c-img2{width:80px}
.c-img3{width:128px}
.c-img4{width:176px}
.c-img6{width:272px}
.c-img12{width:560px}
.c-feed-box .c-img1{width:43px}
.c-feed-box .c-img2{width:90px}
.c-feed-box .c-img3{width:137px}
.c-feed-box .c-img4{width:184px}
.c-feed-box .c-img6{width:278px}
.c-feed-box .c-img12{width:560px}
.c-img-l,.c-img-s,.c-img-v,.c-img-w,.c-img-x,.c-img-y,.c-img-z{height:0;overflow:hidden}
.c-img-s{padding-bottom:100%}
.c-img-l{padding-bottom:133.33333333%}
.c-img-w{padding-bottom:56.25%}
.c-img-x{padding-bottom:75%}
.c-img-y{padding-bottom:66.66666667%}
.c-img-v{padding-bottom:116.66666667%}
.c-img-z{padding-bottom:62.5%}
.c-img-radius{border-radius:6px}
.c-img-radius-s{border-radius:2px}
.c-img-radius-small{border-radius:2px}
.c-img-radius-large{border-radius:12px}
.c-img-radius-middle{border-radius:4px}
.c-img-radius-left{border-top-left-radius:6px;border-bottom-left-radius:6px}
.c-img-radius-right{border-top-right-radius:6px;border-bottom-right-radius:6px}
.c-img-radius-left-s{border-top-left-radius:2px;border-bottom-left-radius:2px}
.c-img-radius-right-s{border-top-right-radius:2px;border-bottom-right-radius:2px}
.c-img-radius-left-l{border-top-left-radius:12px;border-bottom-left-radius:12px}
.c-img-radius-right-l{border-top-right-radius:12px;border-bottom-right-radius:12px}
.c-img-mask{position:absolute;top:0;left:0;z-index:2;width:100%;height:100%;background-image:radial-gradient(circle,rgba(0,0,0,0),rgba(0,0,0,.04));background-image:-ms-radial-gradient(circle,rgba(0,0,0,0),rgba(0,0,0,.04))}
.c-img-border{content:'';position:absolute;top:0;left:0;bottom:0;right:0;border:1px solid rgba(0,0,0,.05)}
.c-img-circle{border-radius:100%;overflow:hidden}
.c-input{display:inline-block;font:13px/23px Arial,sans-serif;color:#333;padding:0 10px;border:1px solid #d7d9e0;border-radius:6px;height:28px;line-height:28px\9;font-size:13px;outline:0;box-sizing:content-box;-webkit-box-sizing:content-box;-moz-box-sizing:content-box;vertical-align:top;overflow:hidden}
.c-input .c-icon{float:right;margin-top:6px;font-size:16px;color:#9195a3}
.c-input .c-icon-left{float:left;margin-right:4px}
.c-input input{float:left;font-size:13px;border:0;outline:0}
.c-input input::-webkit-input-placeholder{color:#9195a3}
.c-input input::-ms-input-placeholder{color:#9195a3}
.c-input input::-moz-placeholder{color:#9195a3}
.c-input::-webkit-input-placeholder{color:#9195a3}
.c-input::-ms-input-placeholder{color:#9195a3}
.c-input::-moz-placeholder{color:#9195a3}
.c-input{width:394px}
.c-input input{width:374px}
.c-input-xmini{width:154px}
.c-input-xmini input{width:134px}
.c-input-mini{width:202px}
.c-input-mini input{width:182px}
.c-input-small{width:346px}
.c-input-small input{width:326px}
.c-input-large{width:442px}
.c-input-large input{width:422px}
.c-input-xlarge{width:730px}
.c-input-xlarge input{width:710px}
.c-input12{width:538px}
.c-input12 input{width:518px}
.c-input20{width:922px}
.c-input20 input{width:902px}
.c-checkbox,.c-radio{display:inline-block;position:relative;white-space:nowrap;outline:0;line-height:1;vertical-align:middle;cursor:pointer;width:16px;height:16px}
.c-checkbox-inner,.c-radio-inner{display:inline-block;position:relative;width:16px;height:16px;line-height:16px;text-align:center;top:0;left:0;background-color:#fff;color:#d7d9e0}
.c-checkbox-input,.c-radio-input{position:absolute;top:0;bottom:0;left:0;right:0;z-index:1;opacity:0;filter:alpha(opacity=0)\9;user-select:none;margin:0;padding:0;width:100%;height:100%;cursor:pointer;zoom:1}
.c-checkbox-inner-i,.c-radio-inner-i{display:none;font-size:16px}
.c-checkbox-inner-bg,.c-radio-inner-bg{font-size:16px;position:absolute;top:0;left:0;z-index:1}
.c-checkbox-checked .c-checkbox-inner-i,.c-radio-checked .c-radio-inner-i{color:#4e71f2;display:inline-block}
.c-textarea{font:13px/23px Arial,sans-serif;color:#333;padding:0 10px;border:1px solid #d7d9e0;border-radius:6px;padding:5px 10px;resize:none;outline:0}
.c-textarea::-webkit-input-placeholder{color:#9195a3}
.c-textarea::-ms-input-placeholder{color:#9195a3}
.c-textarea::-moz-placeholder{color:#9195a3}
.c-icon{font-family:cIconfont!important;font-style:normal;-webkit-font-smoothing:antialiased;-moz-osx-font-smoothing:grayscale}
.c-line-clamp1{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.c-line-clamp2{display:-webkit-box;overflow:hidden;-webkit-line-clamp:2;-webkit-box-orient:vertical}
.c-text{display:inline-block;padding:0 2px;text-align:center;vertical-align:middle;font-style:normal;color:#fff;overflow:hidden;line-height:16px;height:16px;font-size:12px;border-radius:4px;font-weight:200}
a.c-text{text-decoration:none!important}
.c-text-info{padding-left:0;padding-right:0;font-weight:700;color:#2b99ff;vertical-align:text-bottom}
.c-text-info span{padding:0 2px;font-weight:400}
.c-text-important{background-color:#1cb7fd}
.c-text-public{background-color:#4e6ef2}
.c-text-warning{background-color:#f60}
.c-text-prompt{background-color:#ffc20d}
.c-text-danger{background-color:#f73131}
.c-text-safe{background-color:#39b362}
.c-text-mult{padding:0 4px;line-height:18px;height:18px;border-radius:4px;font-weight:400}
.c-text-blue{background-color:#4e6ef2}
.c-text-blue-border{border:1px solid #cbd2ff;padding:0 8px;border-radius:4px;font-weight:400;color:#4e6ef2!important}
.c-text-green{background-color:#39b362}
.c-text-green-border{border:1px solid #c9e7cd;padding:0 8px;border-radius:4px;font-weight:400;color:#39b362!important}
.c-text-red{background-color:#f73131}
.c-text-red-border{border:1px solid #f0c8bd;padding:0 8px;border-radius:4px;font-weight:400;color:#f73131!important}
.c-text-yellow{background-color:#ffc20d}
.c-text-yellow-border{border:1px solid #fcedb1;padding:0 8px;border-radius:4px;font-weight:400;color:#ffc20d!important}
.c-text-orange{background-color:#f60}
.c-text-orange-border{border:1px solid #f8d2b0;padding:0 8px;border-radius:4px;font-weight:400;color:#f60!important}
.c-text-pink{background-color:#fc3274}
.c-text-pink-border{border:1px solid #f6c4d7;padding:0 8px;border-radius:4px;font-weight:400;color:#fc3274!important}
.c-text-gray{background-color:#626675}
.c-text-gray-border{border:1px solid #dbdbdb;padding:0 8px;border-radius:4px;font-weight:400;color:#626675!important}
.c-text-dark-red{background-color:#cc2929}
.c-text-gray-opacity{background-color:rgba(0,0,0,.3)}
.c-text-white-border{border:1px solid rgba(255,255,255,.8);padding:0 8px;border-radius:4px;font-weight:400;color:#fff!important}
.c-text-hot{background-color:#f60}
.c-text-new{background-color:#ff455b}
.c-text-fei{background-color:#fc3200}
.c-text-bao{background-color:#de1544}
.c-text-rec{background-color:#4dadfe}
.c-text-business{background-color:#8399f5}
.c-text-time{background-color:rgba(0,0,0,.3)}
.c-wrapper{word-wrap:break-word;word-break:break-all;font:14px/24px Arial,sans-serif;color:#222}
.c-wrapper:after{display:block;height:0;content:"";clear:both;visibility:hidden}
.c-container{width:560px}
.c-wrapper-l{width:1088px}
.c-wrapper-l .c-container-r{width:368px}
.c-wrapper-s{width:944px}
.c-wrapper-s .c-container-r{width:272px}
@media screen and (max-width:1340px){
.c-wrapper{width:944px}
.c-wrapper .c-container-r{width:272px}
}
.c-wrapper-hot-news-all{width:654px}
#s_main.c-wrapper-hot-news-all{padding-top:9px}
@media screen and (max-width:1340px){
.c-wrapper.c-wrapper-hot-news-all{width:654px}
}
.c-dialog-box{display:none;position:absolute;z-index:999;box-shadow:0 2px 10px 0 rgba(0,0,0,.1);-webkit-box-shadow:0 2px 10px 0 rgba(0,0,0,.1);-moz-box-shadow:0 2px 10px 0 rgba(0,0,0,.1);-o-box-shadow:0 2px 10px 0 rgba(0,0,0,.1);border-radius:16px;background:#fff;padding:19px 24px}
.c-dialog-box .c-dialog-close{position:absolute;cursor:pointer;top:12px;right:12px;height:14px;width:14px;line-height:1;color:#d7d9e0}
.c-dialog-box .c-dialog-close:hover{color:#315efb}
.c-floating-box{background:#fff;box-shadow:0 2px 10px 0 rgba(0,0,0,.15);-webkit-box-shadow:0 2px 10px 0 rgba(0,0,0,.15);-moz-box-shadow:0 2px 10px 0 rgba(0,0,0,.15);-o-box-shadow:0 2px 10px 0 rgba(0,0,0,.15);border-radius:12px;*border:1px solid #d7d9e0}
.c-link{color:#222;text-decoration:none}
.c-link:visited{color:#626675}
.c-link:hover{color:#315efb;text-decoration:none}
.c-capsule-tip{display:inline-block;background:#f63051;border-radius:7px;padding:0 4px;height:13px;font-size:11px;line-height:14px;color:#fff;text-align:center}
.darkmode.c-container{color:#a8acad}
.darkmode .c-abstract{color:#a8acad}
.darkmode .c-link{color:#ffd862}
.darkmode .c-link:hover{color:#fff762}
.darkmode .c-link:visited{color:#ea80ff}
.darkmode .c-btn{background-color:#31313b}
.darkmode .c-btn,.darkmode .c-btn:visited{color:#a8acad!important}
.darkmode .c-btn-primary,.darkmode .c-btn-primary:visited{background:#4e6ef2!important;color:#fff!important}
.darkmode .c-btn-disable{color:#6f7273!important}
.darkmode .c-btn-disable:visited{color:#6f7273!important}
.darkmode .c-btn-disable:hover{color:#6f7273!important;background-color:#31313b}
.darkmode .c-btn-add-disable:hover{color:#6f7273!important;background-color:#31313b}
.darkmode .c-color-link{color:#ffd862}
.darkmode .c-color-visited{color:#ea80ff}
.darkmode .c-color-t{color:#a8acad}
.darkmode .c-color-text{color:#a8acad}
.darkmode .c-color-link{color:#a8acad}
.darkmode .c-color-red{color:#f14d2d}
.darkmode .c-color-red:visited{color:#f14d2d}
.darkmode .c-gray{color:#a8acad}
.darkmode .c-color-gray{color:#a8acad}
.darkmode .c-color-gray2{color:#a8acad}
.darkmode .c-text-danger{background-color:#f14d2d}
.darkmode .c-text-red{background-color:#f14d2d}
.darkmode .c-text-red-border{color:#f14d2d!important}
.darkmode .c-text-public{background-color:#6783f4}
.darkmode .c-text-blue{background-color:#6783f4}
.darkmode .c-text-blue-border{color:#6783f4!important}
.darkmode .c-text-gray{background-color:#a5abac}
.darkmode .c-text-gray-border{color:#a5abac!important}
.darkmode .c-text-dark-red{background-color:#f74a4a}
.darkmode .c-text-bao{background-color:#ff2d8b}
.darkmode .c-capsule-tip{background:#f14d2d}
.darkmode .c-select{color:#a8acad}
.darkmode .c-select-arrow-up{color:#a8acad}
.darkmode .c-select-item:hover{color:#fff762}
.darkmode .c-select-item-selected{color:#fff762}
.darkmode .c-tabs-nav{color:#a8acad}</style><style type="text/css" index="common">#head_wrapper{position:relative;height:40%;min-height:314px;max-height:510px;width:1000px;margin:0 auto}
#head_wrapper.head-board-sample{height:45%;min-height:344px;max-height:510px}
#head_wrapper.head-board-sample .s_form_wrapper #result_logo{margin-left:-30px}
@media screen and (max-width:1340px){
#head_wrapper.head-board-sample .s_form_wrapper a#result_logo{margin-left:-100px}
}
#head_wrapper .s-p-top{height:24vh;min-height:185px;max-height:310px;position:relative;z-index:0;text-align:center}
#head_wrapper #s_lg_img,#head_wrapper #s_lg_img_aging,#head_wrapper #s_lg_img_new{bottom:15px!important}
#head_wrapper input{outline:0;-webkit-appearance:none}
#head_wrapper input::-webkit-input-placeholder{font-size:16px;color:#9195a3}
#head_wrapper .s_btn_wr,#head_wrapper .s_ipt_wr{display:inline-block;*display:inline;zoom:1;background:0 0;vertical-align:top;*vertical-align:middle}
#head_wrapper .s_ipt_wr{position:relative;width:546px}
#head_wrapper .s_btn_wr{width:108px;height:44px;position:relative;z-index:2}
#head_wrapper .s_ipt_wr:hover #kw{border-color:#a7aab5}
#head_wrapper #kw{width:512px;height:16px;padding:12px 16px;font-size:16px;margin:0;vertical-align:top;outline:0;box-shadow:none;border-radius:10px 0 0 10px;border:2px solid #c4c7ce;background:#fff;color:#222;overflow:hidden;box-sizing:content-box;-webkit-tap-highlight-color:transparent}
#head_wrapper #kw:focus{border-color:#4e6ef2!important;opacity:1;filter:alpha(opacity=100)\9}
#head_wrapper .soutu-env-mac .has-voice #kw{width:411px;padding-right:119px}
#head_wrapper.s-down .soutu-env-mac .has-voice #kw{width:411px;padding-right:119px}
#head_wrapper .soutu-env-mac #kw,#head_wrapper .soutu-env-nomac #kw{width:443px;padding-right:87px}
#head_wrapper.s-down .soutu-env-mac #kw,#head_wrapper.s-down .soutu-env-nomac #kw{width:443px;padding-right:87px}
#head_wrapper .soutu-env-mac .sam_search.has-voice #kw{width:393px;padding-right:139px}
#head_wrapper.s-down .soutu-env-mac .sam_search.has-voice #kw{width:393px;padding-right:139px}
#head_wrapper .soutu-env-mac .sam_search #kw,#head_wrapper .soutu-env-nomac .sam_search #kw{width:433px;padding-right:99px}
#head_wrapper.s-down .soutu-env-mac .sam_search #kw,#head_wrapper.s-down .soutu-env-nomac .sam_search #kw{width:433px;padding-right:99px}
#head_wrapper .s_form{width:654px;height:100%;margin:0 auto;text-align:left;z-index:100}
#head_wrapper .s_form .bdsug.bdsug-new ul .bdsug-ai-guide{box-sizing:content-box;display:inline-flex;align-items:center;justify-content:center;position:absolute;top:5.5px;height:16px;background:#6e4BFA17;padding:0 3px;border-radius:3px;margin-left:7px}
#head_wrapper .s_form .bdsug.bdsug-new ul #bdsug-ai-text{color:#6e4bfa;font-size:12px}
#head_wrapper .s_form .bdsug.bdsug-new ul .bdsug-logo{margin-right:3px;height:12px}
#head_wrapper .s_btn{cursor:pointer;width:108px;height:44px;line-height:45px;line-height:44px\9;padding:0;background:0 0;background-color:#4e6ef2;border-radius:0 10px 10px 0;font-size:17px;color:#fff;box-shadow:none;font-weight:400;border:none;outline:0}
#head_wrapper .s_btn:hover{background-color:#4662d9}
#head_wrapper .s_btn:active{background-color:#4662d9}
#head_wrapper.s-upgrade-down{position:fixed;top:0;left:0;height:50px;min-height:50px;z-index:20;width:100%;padding-top:15px;_margin:0 auto}
#head_wrapper.s-upgrade-down .s_form .s_form_wrapper #main-wrapper.two-line-sample #main-input-wrapper{height:42px}
#head_wrapper.s-upgrade-down .s_form .s_form_wrapper #main-wrapper.two-line-sample #main-input-wrapper.wrapper-focus{height:83px}
#head_wrapper.s-upgrade-down .s_form .s_form_wrapper #main-wrapper.two-line-sample #main-input-wrapper #main-ipt.main-input-blur{line-height:22px;max-width:442px}
#head_wrapper.s-upgrade-down .s_form .s_form_wrapper #main-wrapper.two-line-sample #main-input-wrapper #main-ipt.main-input-multi-line{line-height:22px}
#head_wrapper.s-upgrade-down .s_form .s_form_wrapper #main-wrapper.two-line-sample #main-input-wrapper #main-btn{bottom:3px;right:3px}
#head_wrapper.s-upgrade-down .s_form .s_form_wrapper #result_logo{margin-top:4px}
#head_wrapper.s-upgrade-down .s_form .s_form_wrapper .s-p-top{display:none}
#head_wrapper.s-upgrade-down #result_logo{display:inline-block;*display:inline;zoom:1;vertical-align:middle}
@-webkit-keyframes fadein{
from{opacity:0}
to{opacity:1}
}
#head_wrapper.s-upgrade-down #result_logo{-webkit-animation:fadein 1s}
#head_wrapper.s-upgrade-down #result_logo img{width:101px}
#head_wrapper.s-down{position:fixed;_position:static;top:0;left:0;height:50px;min-height:50px;z-index:20;width:100%;padding-top:15px;_margin:0 auto}
#head_wrapper.s-down .s_form{width:100%;min-width:1250px;margin:0 auto;height:100%;padding-left:0;margin-top:0;margin-left:-15px;min-height:0}
#head_wrapper.s-down .s_form .s_form_wrapper{margin:0 auto}
#head_wrapper.s-down .s-p-top{display:none}
#head_wrapper.s-down #result_logo,#head_wrapper.s-down .fm{display:inline-block;*display:inline;zoom:1;vertical-align:middle;margin-left:-119px}
@-webkit-keyframes fadein{
from{opacity:0}
to{opacity:1}
}
#head_wrapper.s-down #result_logo{-webkit-animation:fadein 1s}
#head_wrapper.s-down .fm{margin:0 0 0 16px}
#head_wrapper.s-down #result_logo img{width:101px}
#head_wrapper.s-down #kw{padding:10px 16px;width:516px}
#head_wrapper.s-down .s_ipt_wr{width:546px}
#head_wrapper.s-down .s_btn,#head_wrapper.s-down .s_btn_wr{height:40px}
#head_wrapper.s-down .s_btn{line-height:41px;line-height:40px\9}
#head_wrapper.s-down .soutu-env-newindex .sam_search #kw{padding-top:11px;padding-bottom:11px}
#head_wrapper.s-down .soutu-env-newindex .sam_search .s_btn_wr{height:44px}
#head_wrapper.s-down .soutu-env-newindex .sam_search .s_btn{height:44px;line-height:44px}
#head_wrapper.s-down .soutu-env-newindex .sam_search .sam_search_rec_hover{top:52px}
#head_wrapper .ipt_rec,#head_wrapper .soutu-btn{background:#fff url('https://pss.bdstatic.com/static/superman/img/searchbox/nicon-10750f3f7d.png') no-repeat;width:24px;height:20px}
@media only screen and (-webkit-min-device-pixel-ratio:2){
#head_wrapper .ipt_rec,#head_wrapper .soutu-btn{background-image:url('https://pss.bdstatic.com/static/superman/img/searchbox/nicon-2x-6258e1cf13.png');background-size:24px 96px}
}
#head_wrapper .soutu-btn{background-position:0 -51px;right:16px;margin-top:-9px}
#head_wrapper .soutu-btn:hover{background-position:0 -75px}
#head_wrapper .ipt_rec{background-position:0 -2px;top:50%;right:52px!important;margin-top:-10px}
#head_wrapper .ipt_rec:hover{background-position:0 -26px}
#head_wrapper .ipt_rec:after{display:none}
#head_wrapper.s-down .placeholder{top:10px}
#head_wrapper .placeholder{position:absolute;top:12px;left:18px;right:18px;height:20px;overflow:hidden;pointer-events:none}
#head_wrapper .placeholder .placeholder-ul{position:absolute;left:0;right:0;top:0;transition:top .2s ease-in}
#head_wrapper .placeholder .placeholder-ul li{height:20px;color:#626675;font-size:16px;font-weight:400;line-height:20px}
#head_wrapper .placeholder .placeholder-ul .c-placeholder-bottom{margin-bottom:4px}
#head_wrapper .under-searchbox-tips{font-size:13px;color:#222;text-align:center}
#head_wrapper .under-searchbox-tips .links-link{color:#222;display:inline-block}
#head_wrapper .under-searchbox-tips .links-link:hover{color:#315efb}
#head_wrapper .under-searchbox-tips .links-link--image{display:inline-block;width:176px;height:30px;border-radius:6px;overflow:hidden;margin-top:8px}
#head_wrapper .under-searchbox-tips .links-emphasize-link{margin-top:2px;margin-right:20px;padding:0 8px;font-size:14px;text-decoration:none;line-height:30px;display:inline-block;color:#2027b4;border-radius:6px;background:#f5f7fe}
#head_wrapper .under-searchbox-tips .links-emphasize-link:hover{color:#315efb}
#head_wrapper .under-searchbox-tips .links-emphasize-link.last{margin-right:0}
#head_wrapper .under-searchbox-tips .icon{display:inline-block;background-color:#dadde2;width:4px;height:4px;border-radius:50%;margin:0 20px;line-height:18px;vertical-align:top}
#head_wrapper #m{margin:38px auto 0 auto;width:100%}
#head_wrapper #m .icon{margin-top:8px}
#head_wrapper #s_lm_wrap{position:static;margin:32px auto 0 auto;width:100%}
#head_wrapper #s_lm_wrap .links-wrap{display:inline-block;margin:0 auto}
#head_wrapper #s_lm_wrap .links-wrap .links-link:hover{text-decoration:none}
#head_wrapper #s_lm_wrap .links-wrap .icon{margin-top:13px}
#head_wrapper.s-ps-islite #m{position:absolute;bottom:-56px}
#head_wrapper.s-ps-islite #s_lm_wrap{position:absolute;margin-bottom:0;bottom:-62px;left:0;box-sizing:border-box}
#head_wrapper .quickdelete-wrap{position:relative}
#head_wrapper .quickdelete{width:18px;height:18px;font-size:16px;line-height:18px;text-align:center;position:absolute;top:50%;margin-top:-9px;right:16px;display:none;cursor:pointer}
#head_wrapper .quickdelete-line{display:none;height:14px;width:1px;background-color:#f5f5f6;position:absolute;top:50%;right:0;margin-top:-7px}
#head_wrapper #form.has-voice.fm .quickdelete{right:95px}
#head_wrapper #form.has-voice.fm .quickdelete-line{right:83px}
#head_wrapper #form.has-soutu .quickdelete{right:63px}
#head_wrapper #form.has-soutu .quickdelete-line{right:51px}
#head_wrapper #form.has-voice.sam_search.fm .quickdelete{right:111px}
#head_wrapper #form.has-voice.sam_search.fm .quickdelete-line{right:95px}
#head_wrapper #form.has-soutu.sam_search .quickdelete{right:71px}
#head_wrapper #form.has-soutu.sam_search .quickdelete-line{right:55px}
#head_wrapper .sam_search{border-radius:12px}
#head_wrapper .sam_search #kw{border-radius:12px 10px 10px 12px;border:2px solid #4e6ef2;background-clip:padding-box;-ms-background-clip:padding-box;-webkit-background-clip:padding-box;height:20px;line-height:20px;font-size:18px;padding:12px 14px;width:518px}
#head_wrapper .sam_search .s_ipt_wr{z-index:4}
#head_wrapper .sam_search .s_ipt_wr.iptfocus #kw,#head_wrapper .sam_search .s_ipt_wr.ipthover #kw{border-color:#1d4fff!important}
#head_wrapper .sam_search .s_ipt_wr:hover #kw{border-color:#1d4fff!important}
#head_wrapper .sam_search .s_ipt_wr:hover .s_btn{background-color:#1d4fff}
#head_wrapper .sam_search .s_btn_wr{width:116px;height:48px;margin-left:-8px}
#head_wrapper .sam_search .s_btn_wr .btnfocus.s_btn,#head_wrapper .sam_search .s_btn_wr .btnhover.s_btn{background-color:#1d4fff}
#head_wrapper .sam_search .s_btn{width:116px;height:48px;line-height:48px;z-index:-1;font-size:18px;padding-left:8px;border-radius:0 12px 12px 0;background-color:#4e6ef2}
#head_wrapper .sam_search .s_btn:hover{background-color:#1d4fff}
#head_wrapper .sam_search.sam_form_shadow{box-shadow:0 4px 2px 0 rgba(0,0,0,.1)}
#head_wrapper .sam_search .iptfocus.s_btn_wr #kw{border-color:#1d4fff}
#head_wrapper .sam_search .quickdelete{color:#e4e4e5}
#head_wrapper .sam_search .quickdelete:hover{color:#c4c7ce}
#head_wrapper .sam_search .quickdelete-line{height:24px;margin-top:-12px}
#head_wrapper .sam_search .sam_search_rec{right:54px}
#head_wrapper .sam_search .sam_search_rec_hover{top:56px;right:29px}
#head_wrapper .soutu-env-newindex .sam_search #kw{height:18px;line-height:18px;padding-top:13px;padding-bottom:13px}
.s-skin-hasbg #head_wrapper #form.sam_search #kw{border-color:#4e6ef2}
.s-skin-hasbg #head_wrapper #form.sam_search #kw:hover{border-color:#1d4fff;opacity:1;filter:alpha(opacity=100)\9}
.s-skin-hasbg #head_wrapper #form.sam_search #kw:focus{border-color:#1d4fff!important;opacity:1;filter:alpha(opacity=100)\9}
.s-skin-hasbg #head_wrapper #form.sam_search #kw.new-ipt-focus{border-color:#1d4fff}
.s-skin-hasbg #head_wrapper .s_btn{background:#4e6ef2;color:#fff}
.s-skin-hasbg #head_wrapper .s_btn:hover{background-color:#4662d9}
.s-skin-hasbg #head_wrapper .s_btn:active{background-color:#4662d9}
.s-skin-hasbg #head_wrapper #form #kw{border-color:#4569ff}
.s-skin-hasbg #head_wrapper #form #kw:hover{border-color:#4569ff;opacity:1;filter:alpha(opacity=100)\9}
.s-skin-hasbg #head_wrapper #form #kw:focus{border-color:#4569ff!important;opacity:1;filter:alpha(opacity=100)\9}
.s-skin-hasbg #head_wrapper #form #kw.new-ipt-focus{border-color:#4569ff}
.s-skin-hasbg #head_wrapper #s_lm_wrap{background-image:none;filter:none}
.s-skin-hasbg #head_wrapper #s_lm_wrap .links-wrap{background-color:rgba(255,255,255,.65)!important;padding:0 12px;border-radius:6px}
.s-skin-hasbg #head_wrapper #s_lm_wrap .links-wrap .icon{margin-top:13px}
.s-skin-hasbg #head_wrapper.s-down #form.sam_search #kw{border-color:#4e6ef2;padding-top:11px;padding-bottom:11px;height:18px;line-height:18px}
.s-skin-hasbg #head_wrapper.s-down #form.sam_search #kw:hover{border-color:#1d4fff;opacity:1;filter:alpha(opacity=100)\9}
.s-skin-hasbg #head_wrapper.s-down #form.sam_search #kw:focus{border-color:#1d4fff!important;opacity:1;filter:alpha(opacity=100)\9}
.s-skin-hasbg #head_wrapper.s-down #form.sam_search #kw.new-ipt-focus{border-color:#1d4fff}
.s-skin-hasbg #head_wrapper.s-down #form.sam_search .s_btn_wr{height:44px}
.s-skin-hasbg #head_wrapper.s-down #form.sam_search .s_btn{background:#4e6ef2;color:#fff;height:44px;line-height:44px}
.s-skin-hasbg #head_wrapper.s-down #form.sam_search .s_btn.btnfocus,.s-skin-hasbg #head_wrapper.s-down #form.sam_search .s_btn:active,.s-skin-hasbg #head_wrapper.s-down #form.sam_search .s_btn:hover{background-color:#1d4fff}
.s-skin-hasbg #head_wrapper.s-down #form #kw{border-color:#c4c7ce}
.s-skin-hasbg #head_wrapper.s-down #form #kw:hover{border-color:#a7aab5;opacity:.8;filter:alpha(opacity=80)\9}
.s-skin-hasbg #head_wrapper.s-down #form #kw:focus{border-color:#4e6ef2!important;opacity:1;filter:alpha(opacity=100)\9}
.s-skin-hasbg #head_wrapper.s-down #form #kw.new-ipt-focus{border-color:#4e6ef2}
.s-skin-hasbg #head_wrapper.s-down .s_btn{background:#4e6ef2;color:#fff}
.s-skin-hasbg #head_wrapper.s-down .s_btn:hover{background-color:#4662d9}
.s-skin-hasbg #head_wrapper.s-down .s_btn:active{background-color:#4662d9}
.new_input_superman #head_wrapper{user-select:none;-webkit-user-select:none}
.new_input_superman #s_form_wrapper{user-select:text;-webkit-user-select:text}
.new_input_superman .s_form{transition:margin-left 160ms ease}
.new_input_superman #head_wrapper{user-select:none;-webkit-user-select:none}
.new_input_superman #s_form_wrapper{user-select:text;-webkit-user-select:text}
.new_input_superman .s_form_nologin #main-wrapper{position:absolute;top:60%}
.new_input_superman .s_form_login #s_new_search_guide.two-line-sample .new_search_guide_bub{bottom:-76px}
.new_input_superman #main-wrapper{width:656px;position:relative}
.new_input_superman #main-wrapper #main-input-wrapper{border-width:1.25px;border-color:#b7b9c1;border-style:solid;border-radius:12px;position:relative;transition:height 160ms ease;height:42px;background-color:#fff;z-index:3}
.new_input_superman #main-wrapper #main-input-wrapper:hover{cursor:text}
.new_input_superman #main-wrapper #main-input-wrapper.height-line-1{height:42px}
.new_input_superman #main-wrapper #main-input-wrapper.height-line-2{height:83px}
.new_input_superman #main-wrapper #main-input-wrapper.height-line-3{height:105px!important}
.new_input_superman #main-wrapper #main-input-wrapper.height-line-4{height:127px!important}
.new_input_superman #main-wrapper #main-input-wrapper.height-line-5{height:149px!important}
.new_input_superman #main-wrapper #main-input-wrapper:hover{box-shadow:0 2px 12px 0 rgba(51,91,255,.08)}
.new_input_superman #main-wrapper #main-input-wrapper.wrapper-focus{border-color:#335bff;box-shadow:0 2px 12px 0 rgba(51,91,255,.08)}
.new_input_superman #main-wrapper #main-input-wrapper .main-input{display:inline-block;word-wrap:break-word;min-width:433px;max-width:calc(100% - 4px);max-height:88px;padding-right:14px;padding-left:16px;margin-top:12px;margin-bottom:12px;font-size:16px;line-height:18px;box-sizing:border-box;vertical-align:middle;overflow-y:hidden;outline:0;cursor:text;color:#333;caret-color:#335bff}
.new_input_superman #main-wrapper #main-input-wrapper .main-input::-webkit-scrollbar{display:initial;width:4px}
.new_input_superman #main-wrapper #main-input-wrapper .main-input::-webkit-scrollbar-track{background:0 0}
.new_input_superman #main-wrapper #main-input-wrapper .main-input::-webkit-scrollbar-thumb{background:rgba(51,51,51,.2);border-radius:2px}
.new_input_superman #main-wrapper #main-input-wrapper .main-input-multi-line{margin-top:10px;line-height:22px;overflow-y:scroll;scroll-behavior:smooth}
.new_input_superman #main-wrapper #main-input-wrapper .main-input-blur{line-height:18px;overflow:hidden;white-space:nowrap;text-overflow:ellipsis;max-width:433px}
.new_input_superman #main-wrapper #main-input-wrapper .main-input:empty::before{content:'请输入你的问题';position:absolute;top:12px;left:16px;color:#aaa;pointer-events:none}
.new_input_superman #main-wrapper #main-input-wrapper .main-input:not(:empty)::before{content:""}
.new_input_superman #main-wrapper #main-input-wrapper .main-submit{position:absolute;border-radius:0 12px 12px 0;right:-1px;bottom:-1px;height:44px;width:112px;background-color:#335bff;transition:all 160ms ease}
.new_input_superman #main-wrapper #main-input-wrapper .main-submit:before{content:"";position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);background-image:url(https://gips0.baidu.com/it/u=3785693112,2605121048&fm=3028&app=3028&f=PNG&fmt=auto&q=75&size=f189_45);background-size:cover;width:63px;height:15px}
.new_input_superman #main-wrapper #main-input-wrapper .main-submit:hover{background-color:#2949cc;cursor:pointer}
.new_input_superman #main-wrapper #main-input-wrapper .main-submit-multi-line{width:108px;height:36px;border-radius:9px;bottom:3px;right:3px}
.new_input_superman #main-wrapper #main-input-wrapper .main-tool{position:absolute;bottom:10px;right:112px;display:flex}
.new_input_superman #main-wrapper #main-input-wrapper .main-tool #pic-mic{display:flex}
.new_input_superman #main-wrapper #main-input-wrapper .main-tool #pic-mic .ipt_rec{position:relative;margin-top:0;height:21px;width:21px;margin-right:15px;top:0;right:0!important;left:0;background-image:url('https://pss.bdstatic.com/static/superman/img/searchbox/voice-e255e654cb.png');background-size:21px 42px;background-repeat:no-repeat;background-position:0 0}
.new_input_superman #main-wrapper #main-input-wrapper .main-tool #pic-mic .ipt_rec:hover{background-position:0 -21px}
.new_input_superman #main-wrapper #main-input-wrapper .main-tool #pic-mic .soutu-btn{position:relative;display:block;top:-.5px;right:0;margin-top:1px;margin-right:17px;height:21px;width:21px;background-image:url('https://pss.bdstatic.com/static/superman/img/searchbox/camera-4683ae6561.png');background-size:21px 42px;background-repeat:no-repeat;background-position:0 0}
.new_input_superman #main-wrapper #main-input-wrapper .main-tool #pic-mic .soutu-btn:hover{cursor:pointer;background-position:0 -21px}
.new_input_superman #main-wrapper #main-input-wrapper .main-tool .clear-btn{display:none;width:21px;height:21px;background-image:url('https://pss.bdstatic.com/static/superman/img/searchbox/clear-48eb6e3446.png');background-size:21px 42px;background-position:0 0;background-repeat:no-repeat;margin-right:15px;margin-top:.5px}
.new_input_superman #main-wrapper #main-input-wrapper .main-tool .clear-btn:hover{cursor:pointer;background-position:0 -21px}
.new_input_superman #main-wrapper #main-input-wrapper .voice-hover{right:40px;top:40px}
.new_input_superman #main-wrapper #main-input-wrapper .soutu-hover-tip{background:#fff;box-shadow:0 2px 10px 0 rgba(0,0,0,.1);border-radius:6px;height:32px;width:94px;text-align:center;line-height:32px;font-size:14px;color:#626675;position:absolute;z-index:2;top:40px;right:0}
.new_input_superman #main-wrapper.two-line-sample #main-input-wrapper{height:83px}
.new_input_superman #main-wrapper.two-line-sample #main-input-wrapper .main-input{margin-top:10px;overflow-y:scroll;line-height:22px}
.new_input_superman #main-wrapper.two-line-sample #main-input-wrapper .main-input:empty::before{top:10.5px}
.new_input_superman #main-wrapper.two-line-sample #main-input-wrapper .main-input-blur{max-width:100%;height:22px;overflow:hidden}
.new_input_superman #main-wrapper.two-line-sample #main-input-wrapper .main-submit{width:108px;height:36px;border-radius:9px;bottom:3px;right:3px}
.new_input_superman #main-wrapper .soutu-layer{border-radius:12px;width:656px}
.new_input_superman #main-wrapper .soutu-layer .soutu-url{background-color:#fff;border:1.25px solid #335bff;border-radius:12px;box-shadow:0 2px 12px 0 rgba(51,91,255,.08)}
.new_input_superman #main-wrapper .soutu-layer .soutu-url .soutu-url-wrap{border-radius:12px;height:42px;width:calc(100% - 115px)}
.new_input_superman #main-wrapper .soutu-layer .soutu-url .soutu-url-wrap #soutu-url-kw{scrollbar-width:thin;scrollbar-color:rgba(51,51,51,.2) transparent;border:unset;width:100%;caret-color:#335bff;font-weight:400}
.new_input_superman #main-wrapper .soutu-layer .soutu-url .soutu-url-wrap #soutu-url-kw::-webkit-scrollbar{width:4px}
.new_input_superman #main-wrapper .soutu-layer .soutu-url .soutu-url-wrap #soutu-url-kw::placeholder{color:#9195a3}
.new_input_superman #main-wrapper .soutu-layer .soutu-url .soutu-url-wrap #soutu-url-kw::-webkit-input-placeholder{color:#9195a3}
.new_input_superman #main-wrapper .soutu-layer .soutu-url .soutu-url-wrap #soutu-url-kw::-moz-placeholder{color:#9195a3}
.new_input_superman #main-wrapper .soutu-layer .soutu-url .soutu-url-btn{height:42px;width:112px;background-color:#335bff;border-radius:0 12px 12px 0;position:absolute;right:-1px}
.new_input_superman #main-wrapper .soutu-layer .soutu-url .soutu-url-btn:before{content:"";position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);background-image:url(https://gips0.baidu.com/it/u=3785693112,2605121048&fm=3028&app=3028&f=PNG&fmt=auto&q=75&size=f189_45);background-size:cover;width:63px;height:15px}
.new_input_superman #main-wrapper .soutu-layer.new-wrapper .soutu-url{background-color:#fff}
.new_input_superman #main-wrapper .soutu-layer.new-wrapper .soutu-url .soutu-url-wrap{width:100%;padding-top:9px}
.new_input_superman #main-wrapper .soutu-layer.new-wrapper .soutu-url .soutu-url-wrap #soutu-url-kw{padding-top:0;padding-bottom:0;height:calc(100% - 12px)}
.new_input_superman #main-wrapper .soutu-layer.new-wrapper .soutu-url .soutu-url-btn{position:absolute;bottom:3px;width:108px;height:36px;right:3px;border-radius:9px}
.new_input_superman #head_wrapper.s-ps-islite #main-wrapper{top:calc(100% - 44px)}
.new_input_superman #head_wrapper.s-ps-islite #main-wrapper .bdsug-new{position:absolute}
.new_input_superman #head_wrapper.s-ps-islite #s_new_search_guide{padding-top:0}
.new_input_superman #head_wrapper.s-ps-islite #s_new_search_guide.two-line-sample .new_search_guide_bub{bottom:-77px}
.new_input_superman #head_wrapper:not(.s-ps-islite) .s_form_nologin #s_new_search_guide{position:relative;bottom:-13px;left:50%;transform:translateX(-50%)}
.new_input_superman #head_wrapper #main-wrapper.two-line-sample #s_new_search_guide{top:86px}
.new_input_superman #head_wrapper #main-wrapper #s_new_search_guide{position:absolute;top:44px;left:50%;transform:translateX(-50%)}
.new_input_superman #head_wrapper #main-wrapper #s_new_search_guide .new_search_guide_bub{position:unset;transform:unset}
.useAriaTheme .new_input_superman #main-wrapper #main-input-wrapper .main-tool #pic-mic .ipt_rec{background-image:url('https://pss.bdstatic.com/static/superman/img/searchbox/voicedark-df27dd3f94.png')}
.useAriaTheme .new_input_superman #main-wrapper #main-input-wrapper .main-tool #pic-mic .soutu-btn{background-image:url('https://pss.bdstatic.com/static/superman/img/searchbox/cameradark-ff227fa386.png')}
#head_wrapper.s-recommend-style .s_ipt_wr{position:relative;width:593px}
#head_wrapper.s-recommend-style .s_form{width:701px}
#head_wrapper.s-recommend-style .s_btn_wr{height:51px}
#head_wrapper.s-recommend-style .s_btn{height:51px}
#head_wrapper.s-recommend-style #kw{height:23px;width:559px}
#head_wrapper.s-recommend-style .soutu-env-mac .has-voice #kw{width:458px}
#head_wrapper.s-recommend-style .soutu-env-mac #kw,#head_wrapper.s-recommend-style .soutu-env-nomac #kw{width:490px}
#head_wrapper.s-recommend-style .soutu-env-mac #kw,#head_wrapper.s-recommend-style .soutu-env-nomac #kw{width:490px}
#head_wrapper.s-recommend-style #form .bdsug-new{width:591px;top:42px}
#head_wrapper.s-recommend-style .soutu-env-mac .sam_search.has-voice #kw{width:440px}
#head_wrapper.s-recommend-style .soutu-env-mac .sam_search #kw,#head_wrapper.s-recommend-style .soutu-env-nomac .sam_search #kw{width:480px}
#head_wrapper.s-recommend-style .soutu-layer.new-pmd{width:701px}
#head_wrapper.s-recommend-style .soutu-url{height:51px}
#head_wrapper.s-recommend-style .soutu-env-new .soutu-layer .soutu-url-wrap{width:593px;height:51px}
#head_wrapper.s-recommend-style .soutu-env-new .soutu-layer #soutu-url-kw{width:563px;height:29px}
#head_wrapper.s-recommend-style .soutu-url-btn.soutu-url-btn-new{height:51px;line-height:51px}
#head_wrapper.s-recommend-style.s-down .s_ipt_wr{width:546px;height:40px}
#head_wrapper.s-recommend-style.s-down .s_form{width:100%}
#head_wrapper.s-recommend-style.s-down .s_btn_wr{height:40px}
#head_wrapper.s-recommend-style.s-down .s_btn{height:40px}
#head_wrapper.s-recommend-style.s-down #kw{height:16px;width:512px}
#head_wrapper.s-recommend-style.s-down #form .bdsug-new{width:544px;top:31px}
#head_wrapper.s-recommend-style.s-down .soutu-env-mac .has-voice #kw{width:411px}
#head_wrapper.s-recommend-style.s-down .soutu-env-mac .has-voice #kw{width:411px}
#head_wrapper.s-recommend-style.s-down .soutu-env-mac #kw,#head_wrapper.s-recommend-style.s-down .soutu-env-nomac #kw{width:443px}
#head_wrapper.s-recommend-style.s-down .soutu-env-mac .sam_search #kw,#head_wrapper.s-recommend-style.s-down .soutu-env-nomac .sam_search #kw{width:433px}
#head_wrapper.s-recommend-style.s-down .soutu-env-mac .sam_search.has-voice #kw{width:393px}
#head_wrapper.s-recommend-style .soutu-env-newindex .sam_search .s_btn_wr{height:40px}
#head_wrapper.s-recommend-style .soutu-env-newindex .sam_search .s_btn{height:40px}
#head_wrapper.s-recommend-style.s-down .soutu-layer.new-pmd{width:654px}
#head_wrapper.s-recommend-style.s-down .soutu-url{height:40px}
#head_wrapper.s-recommend-style.s-down .soutu-env-new .soutu-layer .soutu-url-wrap{width:546px;height:40px}
#head_wrapper.s-recommend-style.s-down .soutu-env-new .soutu-layer #soutu-url-kw{width:512px;height:18px}
#head_wrapper.s-recommend-style.s-down .soutu-url-btn.soutu-url-btn-new{height:40px;line-height:40px}
#head_wrapper.s-recommend-style .ipt_rec,#head_wrapper.s-recommend-style .soutu-btn{background-image:url(https://gips2.baidu.com/it/u=3220710717,862447296&fm=3028&app=3028&f=PNG&fmt=auto&q=75&size=f48_192);background-size:24px 96px}
#s_main.main.s-recommend-style-font .s-news-item-title{font-size:18px}
#s_main.main.s-recommend-style-font .s-menu-gurd{font-size:16px}
#s_wrap #recommend_tool_list{width:1100px;position:relative;margin-left:auto;margin-right:auto;margin-bottom:90px;margin-top:calc(44px + 30px - max(16vh,126px))}
.s-recommend-style+#s_wrap #recommend_tool_list{margin-top:calc(51px + 30px - max(16vh,126px))}
#s_wrap #recommend_tool_list.login{margin-top:calc(44px + 30px - max(16vh,126px));margin-bottom:76px}
.s-recommend-style+#s_wrap #recommend_tool_list.login{margin-top:calc(51px + 30px - max(16vh,126px))}
#s_wrap .s-hot-board-recommend-style{width:1100px;position:relative;margin:0 auto}
#head_wrapper.s-recommend-style.s-ps-islite+#s_wrap #recommend_tool_list{margin:-340px auto 0 auto}
.s-skin-hasbg #s_wrap #recommend_tool_list{margin-top:-60px;margin-bottom:50px}
.s-skin-hasbg #s_wrap #recommend_tool_list{width:1150px}
@media screen and (max-width:1340px){
.s-skin-hasbg #s_wrap #recommend_tool_list{width:1008px}
}
@media screen and (max-width:1340px){
#s_wrap #recommend_tool_list,#s_wrap .s-hot-board-recommend-style{width:944px}
}
#s_top_wrap{position:absolute;z-index:99;min-width:1000px;width:100%}
#s_top_wrap.s-down,#s_top_wrap.s-upgrade-down{position:fixed;_position:absolute;top:0;left:0;height:74px;z-index:10;width:100%}
#s_top_wrap .s-center-box{position:relative;z-index:1;width:100%;_width:1000px;height:100%}
#s_top_wrap.s-down .s-center-box{box-shadow:0 2px 10px 0 rgba(0,0,0,.1);background-color:#fff;border-bottom:1px solid #888\9;_border-bottom:0}
#s_top_wrap.s-upgrade-down .s-center-box{box-shadow:0 2px 10px 0 rgba(0,0,0,.1);background-color:#fff;border-bottom:1px solid #888\9;_border-bottom:0}
#s_top_wrap .s-top-nav{position:absolute;top:70px;width:100%;min-width:1250px;_width:1000px;height:40px;overflow:hidden;display:none}
#s_top_wrap .s-top-nav .s-menus-outer{margin-left:-25px}
#s_top_wrap.s-down.sam_top_wrap{height:74px}
#s_top_wrap.sam_top_wrap .s-top-nav{top:74px}
:root{--bubble-width:500px;--bubble-padding-left:500px}
.s-top-wrap{border-bottom:0;height:60px;background:#fff}
.s-top-left-new{position:absolute;left:0;top:0;z-index:100;height:60px;padding-left:24px}
.s-top-left-new .mnav{margin-right:24px;margin-top:19px;display:inline-block;position:relative}
.s-top-left-new .mnav:has(.s-top-ai-tab-container){margin-top:-5px;vertical-align:middle}
.s-top-left-new .mnav:hover .s-bri,.s-top-left-new a:hover{color:#315efb;text-decoration:none}
.s-top-left-new .s-top-more-btn{padding-bottom:19px}
.s-top-left-new .s-top-more{display:none;position:absolute;top:29px;right:-12px;width:228px;height:290px;padding-top:8px;background:#fff;box-shadow:0 2px 10px 0 rgba(0,0,0,.15);border-radius:12px}
.s-top-left-new .s-top-more .s-top-more-content{float:left;width:60px}
.s-top-left-new .s-top-more .s-top-more-content:nth-child(3n+1){margin-left:8px}
.s-top-left-new .s-top-more .s-top-more-content:nth-child(3n+1),.s-top-left-new .s-top-more .s-top-more-content:nth-child(3n+2){margin-right:16px}
.s-top-left-new .s-top-more .s-top-more-content .img-spacing{margin-right:16px}
.s-top-left-new .s-top-more .s-top-more-content a{width:60px;height:82px;float:left}
.s-top-left-new .s-top-more .s-top-more-content img{width:42px;height:42px;margin-top:8px;margin-left:8px;margin-right:8px;border:1px solid rgba(0,0,0,.03);border-radius:8px;display:block}
.s-top-left-new .s-top-more .s-top-more-content .s-top-more-title{width:60px;text-align:center;margin-top:3px}
.s-top-left-new .s-top-more .s-top-more-content>a:hover{background-color:#f1f3fd;border-radius:8px}
.s-top-left-new .s-top-more .s-top-more-content>a:hover .s-top-more-title{color:#315efb}
.s-top-left-new .s-top-more .s-top-tomore{width:212px;height:27px;position:absolute;bottom:10px;left:8px;padding-top:5px}
.s-top-left-new .s-top-more .s-top-tomore:hover{background-color:#f1f3fd;border-radius:8px}
.s-top-left-new .s-top-more .s-top-tomore:hover a{color:#315efb}
.s-top-left-new .s-top-more-btn:hover .s-top-more{display:block}
.s-top-left-new .s-top-tab-image{height:14px;margin-bottom:-2px}
.s-top-left-new .s-top-ai-tab-container{position:relative;overflow:hidden;font-family:PingFang SC;height:30px;background:linear-gradient(103deg,rgba(229,240,255,.7) 9.04%,rgba(237,238,255,.7) 89.22%);border-radius:20px;padding:0 12px;display:flex;flex-direction:row;align-items:center;gap:3px}
.s-top-left-new .s-top-ai-tab-container .s-top-tab-image{margin:0 auto}
.s-top-left-new .s-top-ai-tab-container .s-top-aitab-text{font-size:14px;font-weight:500;font-family:PingFang SC;background:#6e4bfa;background-clip:text;-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.s-top-left-new .s-top-ai-tab-container .s-top-aitab-arrow{margin-left:-3px;color:#6e4bfa}
.s-top-left-new .s-top-ai-tab-container .s-top-aitab-animation{position:absolute;top:0;left:calc(-100% - 40px);height:100%;width:100%;background:linear-gradient(to right,rgba(102,212,255,.4),rgba(255,117,223,.3));background-size:100%;animation:topbar_aitab_shine 1s linear;animation-delay:1s;pointer-events:none;filter:blur(20px)}
@keyframes topbar_aitab_shine{
0%{left:calc(-100% - 40px)}
100%{left:calc(100% + 40px)}
}
.s-top-left-new .s-top-ai-tab-text-container{position:relative}
.s-top-left-new .s-top-ai-tab-text-container .s-top-ai-tab-text-animation{position:absolute;top:50%;transform:translateY(-50%);left:0;height:14px;width:100%;margin-top:-.5px;background:linear-gradient(to right,rgba(43,206,255,.8),rgba(208,97,255,.8));background-size:200px;background-position:100%;animation:topbar_aitab_text_shine 1s linear;animation-delay:1s;pointer-events:none;mask-size:100%;-webkit-mask-size:100%;opacity:0}
@keyframes topbar_aitab_text_shine{
0%{background-position:100%;opacity:0}
20%{opacity:1}
80%{opacity:1}
100%{background-position:0;opacity:0}
}
.s-top-right{position:absolute;right:0;top:0;z-index:100;height:60px;padding-right:24px;padding-left:200px;-webkit-tap-highlight-color:transparent}
.s-top-right .s-top-right-text{margin-left:32px;margin-top:19px;display:inline-block;position:relative;vertical-align:top;cursor:pointer}
.s-top-right .s-top-right-text:hover{color:#315efb}
.s-top-right .s-top-username{margin-left:32px;margin-top:15px;display:inline-block;height:30px;position:relative}
.s-top-right .s-top-username .s-top-img-wrapper{position:relative;width:28px;height:28px;border:1px solid #4e71f2;display:inline-block;border-radius:50%}
.s-top-right .s-top-username img{padding:2px;width:24px;height:24px;border-radius:50%}
.s-top-right .s-top-username:hover .user-name{color:#315efb}
.s-top-right .s-top-username .user-name{display:inline-block;max-width:100px;overflow:hidden;white-space:nowrap;text-overflow:ellipsis;-o-text-overflow:ellipsis;vertical-align:top;margin-top:3px;margin-left:6px}
.s-top-right .s-top-username.s-hasmsg-tip .s-top-img-wrapper::after{content:'';position:absolute;top:-1px;right:0;width:6px;height:6px;border:1px solid #fff;border-radius:6px;background:#f63051}
.s-top-right .s-risk-prompt-top{padding-bottom:25px}
.s-top-right .s-risk-prompt-top:hover .s-risk-prompt-content{display:block;cursor:default}
.s-top-right .s-risk-prompt-top:hover .s-risk-prompt-content .s-prompt-btn{cursor:pointer}
.s-top-right .s-risk-prompt-top:hover .c-color-t{color:#315efb}
.s-top-right .s-risk-prompt-top .s-risk-prompt-content{display:none;width:290px;padding:10px;top:27px;left:0;position:absolute;float:right;z-index:999;border-radius:12px;box-shadow:0 2px 10px 0 rgba(0,0,0,.1);text-align:left;color:var(--cos-color-text-tiny,#50525c)}
.s-top-right .s-risk-prompt-top .s-risk-prompt-content .s-prompt-btn{text-decoration:underline!important;color:#4367b4}
.s-top-right .s-risk-prompt-top .s-risk-prompt-content .s-prompt-btn:hover{color:#315efb}
.s-top-right .s-risk-prompt-top .s-risk-prompt-content-nologin{left:-99px}
.s-top-right .s-risk-prompt-top .s-risk-prompt-content-nologin-EN{left:-25px}
.s-top-right .s-top-right-new{margin-left:24px}
.s-top-right .s-top-login-btn{display:inline-block;margin-top:-24px;margin-left:32px;font-size:13px}
.s-top-right .s-top-login-btn.btn-fixed{margin-top:18px}
.s-top-right a:hover{text-decoration:none}
.s-top-right .yiyan-entry-right{position:relative;display:inline-block;width:146px;height:24px;top:-2px;margin-right:-8px;cursor:pointer;background-size:contain;background-repeat:no-repeat;background-position:right}
.s-top-right .yiyan-entry-right-nologin{top:18px;margin-right:0}
.s-top-right .operate-wrapper{display:inline-block;position:relative;margin-top:16px;margin-right:-11px}
.s-top-right .operate-wrapper .operate-animate{position:absolute;display:inline-block;top:0;right:0}
.s-top-right .operate-wrapper .operate-image{position:absolute;top:0;right:0;background-repeat:no-repeat;background-position:right;background-size:contain}
.s-top-userset-menu{display:none;width:84px;padding:8px 0;top:48px;position:absolute;right:10px;float:right;z-index:999;text-align:left}
.s-top-userset-menu a{display:block;margin:3px 16px 3px 16px;color:#333}
.s-top-userset-menu a:hover{color:#315efb;text-decoration:none}
.s-top-userset-menu .split-line{display:block;margin:8px 16px;background:#d7d9e0;height:1px}
.s-top-userset-menu .s-msg-count{display:none;margin-left:4px}
.s-top-userset-menu .hide-feed{display:inline-block}
.s-top-userset-menu .show-feed{display:none}
.s-top-userset-menu.hiding-feed .hide-feed{display:none}
.s-top-userset-menu.hiding-feed .show-feed{display:inline-block}
.s-top-userset-menu.hiding-feed .set-close-homepage-tts,.s-top-userset-menu.hiding-feed .set-open-homepage-tts{display:none}
.s-skin-hasbg .s-top-wrap{background:rgba(0,0,0,.2)}
.s-skin-hasbg .s-top-left .mnav,.s-skin-hasbg .s-top-left .mnav .s-bri{color:rgba(255,255,255,.85)}
.s-skin-hasbg .s-top-left .mnav:hover,.s-skin-hasbg .s-top-left .mnav:hover .s-bri{color:#fff;text-decoration:none}
.s-skin-hasbg .s-top-left-new .mnav,.s-skin-hasbg .s-top-left-new .mnav .s-bri{color:rgba(255,255,255,.85)}
.s-skin-hasbg .s-top-left-new .mnav:hover,.s-skin-hasbg .s-top-left-new .mnav:hover .s-bri{color:#fff;text-decoration:none}
.s-skin-hasbg .s-top-right .s-top-right-text.c-color-t{color:rgba(255,255,255,.85)}
.s-skin-hasbg .s-top-right .s-top-right-text.c-color-t:hover{color:#fff}
.s-skin-hasbg .s-top-right .s-top-username .user-name{color:rgba(255,255,255,.85)}
.s-skin-hasbg .s-top-right .s-top-username:hover .user-name{color:#fff}
.s-skin-hasbg .s-top-right .s-top-prompt-content{background-color:#fff}
.s-skin-hasbg .s-top-right .s-top-prompt{color:rgba(255,255,255,.85)}
.s-skin-hasbg .s-top-right .s-top-prompt:hover{color:#315efb}
.s-top-right.s-down{position:fixed;left:0;top:5px;min-width:1250px;width:100%;height:0;text-align:right;padding-right:0}
.s-top-right.s-down>*{display:none}
.s-top-right.s-down>#s-riskprompt-top,.s-top-right.s-down>#s-top-username,.s-top-right.s-down>#s-usersetting-top{display:inline-block}
.s-top-right.s-down #s-top-username{margin-right:24px}
.s-top-right.s-down .s-top-right-text.c-color-t{color:#222}
.s-top-right.s-down .s-top-right-text.c-color-t:hover{color:#315efb}
.s-top-right.s-down .s-top-username .user-name{color:#222}
.s-top-right.s-down .s-top-username:hover .user-name{color:#315efb}
.guide-info{background-color:#fff;box-shadow:0 2px 10px 0 rgba(0,0,0,.1);border-radius:12px 2px 12px 12px;height:36px;width:258px;text-align:left;position:absolute;margin-top:6px;padding:5px 0 5px 10px;display:none}
.guide-info .guide-icon{color:#4e6ef2;font-size:15px;display:inline-block;line-height:36px;vertical-align:top;margin-right:6px}
.guide-info span{display:inline-block;line-height:36px;vertical-align:top;font-size:13px;font-family:Arial,sans-serif;color:#333}
.guide-info .guide-close{color:#c4c7ce;margin-left:11px;display:inline-block;width:20px;height:36px;text-align:center;line-height:36px;vertical-align:top;font-size:13px;cursor:pointer}
.guide-info .guide-close:hover{color:#4e6ef2}
.guide-info-login{width:244px}
.s-ie8-hack .s-top-userset-menu{margin-right:-20px}
.s-ie8-hack .s-top-userset-menu{border:1px solid #f5f5f6}
.sui-wraper .sui-dialog.sui-dialog-homepage-tts-pannel{position:fixed;_position:absolute;margin:-100px 0 0 -200px;border-radius:8px;background:#626675;border:0}
.sui-wraper .sui-dialog.sui-dialog-homepage-tts-pannel .homepage-tts-pannel-text{color:#fff;position:absolute;left:11px;top:8px}
#bottom_layer{visibility:hidden;width:3000px;position:fixed;z-index:302;bottom:0;left:0;height:39px;padding-top:1px;zoom:1;margin:0;line-height:39px;background:#fff}
#bottom_layer .lh{display:inline-block;margin-right:14px}
#bottom_layer .lh .emphasize{text-decoration:underline;font-weight:700}
#bottom_layer .lh:last-child{margin-left:-2px;margin-right:0}
#bottom_layer .lh.activity{font-weight:700;text-decoration:underline}
#bottom_layer a{font-size:12px;text-decoration:none}
#bottom_layer .text-color{color:#bbb}
#bottom_layer .aria-img{width:49px;height:20px;margin-bottom:-5px}
#bottom_layer a:hover{color:#222}
#bottom_layer .s-bottom-layer-content{margin:0 17px;text-align:center}
#bottom_layer .s-bottom-layer-content .auto-transform-line{display:inline}
#bottom_layer .s-bottom-layer-content .auto-transform-line:first-child{margin-right:14px}
.s-bottom-space{position:static;width:100%;height:40px;margin:23px auto 12px}
#bottom_layer .open-content-info a:hover{color:#333}
#bottom_layer .open-content-info .text-color{color:#626675}
.accessibility-icon{margin-left:3px;margin-right:9px;position:relative;display:inline-block;width:20px}
.accessibility-icon .text-color{color:#626675}
.accessibility-icon>span{font-size:15px;color:#aaa}
.open-content-info{position:relative;display:inline-block;width:20px}
.open-content-info>span{cursor:pointer;font-size:14px}
.open-content-info>span:hover{color:#333}
.open-content-info .tip-hover-panel{position:absolute;display:none;padding-bottom:18px}
.open-content-info .tip-hover-panel .rest_info_tip{max-width:560px;padding:8px 12px 8px 12px;background:#fff;border-radius:6px;border:1px solid rgba(0,0,0,.05);box-shadow:0 2px 4px 0 rgba(0,0,0,.1);text-align:left}
.open-content-info .tip-hover-panel .rest_info_tip .tip-wrapper{white-space:nowrap;line-height:20px}
.open-content-info .tip-hover-panel .rest_info_tip .tip-wrapper .tip-item{height:20px;line-height:20px}
.open-content-info .tip-hover-panel .rest_info_tip .tip-wrapper .tip-item:last-child{margin-right:0}
@media screen and (max-width:515px){
.open-content-info{width:16px}
.open-content-info .tip-hover-panel{right:-40px!important}
.accessibility-icon{margin-right:11px}
}
#blind-box{position:fixed;right:24px;bottom:185px;height:44px;width:44px;border-radius:22px}
#blind-box .blind-search-box{position:absolute;bottom:-17px;right:0;min-width:208px;height:80px;box-sizing:border-box;padding:17px 0 17px 6px;overflow:hidden;text-align:right}
#blind-box .blind-search-box .blind-search-area{background-color:#fff;text-align:left;display:inline-block;height:46px;max-width:100%;width:fit-content;white-space:nowrap;overflow:hidden;box-sizing:border-box;padding-right:56px;border-radius:12px 2px 12px 12px;font-size:13px;cursor:pointer;color:#333;transform:translateX(110%);line-height:46px;position:relative}
#blind-box .blind-search-box .blind-search-area .blind-text,#blind-box .blind-search-box .blind-search-area .i{display:inline-block;vertical-align:top}
#blind-box .blind-search-box .blind-search-area i{color:#4e71f2;font-size:14px;margin-left:10px}
#blind-box .blind-search-box .blind-search-area .blind-text{height:100%;min-width:50px;position:relative;line-height:46px;transition:all .3s;overflow:hidden}
#blind-box .blind-search-box .blind-search-area .blind-text:hover{color:#315efb}
#blind-box .blind-search-box .blind-search-area .blind-text .blind-span{line-height:46px;position:absolute;white-space:nowrap;left:0;top:0;opacity:0;transition:all .3s}
#blind-box .blind-search-box .blind-search-area .blind-text .span-now{opacity:1}
#blind-box .blind-search-box .blind-search-area .blind-text .span-next{opacity:0;transform:translateX(-40%)}
#blind-box .blind-search-box .blind-search-area .blind-text .span-last{opacity:0;transform:translateX(40%)}
#blind-box .blind-search-box .blind-box-hover{transform:translateX(0);box-shadow:0 2px 10px 0 rgba(0,0,0,.1)}
#blind-box .blind-search-img{height:80px;width:80px;position:absolute;left:-12px;bottom:0;transform-origin:bottom center;transform:scale(.55);border-radius:40px;cursor:pointer;background-color:#fbfbfb;overflow:hidden}
#blind-box .blind-search-img .blind-img{height:100%;width:100%;position:absolute;top:0;left:0;object-fit:contain;object-position:center;opacity:0;transition:all .3s}
#blind-box .blind-search-img .blind-img-show{opacity:1}
#blind-box .blind-img-hover{transform:scale(1) translateZ(0);border-radius:0;background-color:transparent}
#blind-box .blind-img-ie,#blind-box .blind-title-ie{cursor:pointer;position:absolute;bottom:0;right:0}
#blind-box .blind-title-ie{background-color:#fff;height:46px;width:191px;line-height:46px;box-sizing:border-box;padding:0 10px;font-size:13px;color:#333;visibility:hidden}
#blind-box .blind-title-ie:hover{color:#315efb}
#blind-box .blind-img-ie{height:44px;width:44px;object-fit:cover;object-position:center}
#blind-box:hover .blind-img-ie{height:80px;width:80px;right:-24px}
#blind-box:hover .blind-img-ie,#blind-box:hover .blind-title-ie{visibility:visible}
@media screen and (max-width:1158px){
#blind-box{display:none}
}
#s_popup_advert{position:absolute}
#s_popup_advert .popup-advert{display:none;position:fixed;right:0;bottom:0;z-index:303;width:100%;text-align:center}
#s_popup_advert .advert-link{display:block;width:100%}
#s_popup_advert .advert{display:block;width:100%;height:auto}
#s_popup_advert .right-wrap{position:absolute;right:24px;top:0;width:152px;height:30px;border-radius:6px;line-height:30px;font-size:13px;color:#9195a3}
#s_popup_advert .popup-count-down{float:left;padding-left:10px}
#s_popup_advert .close-wrap{float:right;padding-left:10px;padding-right:10px;cursor:pointer}
#s_popup_advert .close-icon{vertical-align:middle;color:#c0c2c8}
#s_popup_advert .close-text{padding-left:8px}
#s_popup_advert .close-wrap:hover .close-icon{color:#9195a3}
#s_popup_advert .close-wrap:hover .close-text{color:#626675}
#s_popup_advert .advert-shrink{transform:scale(0);-ms-transform:scale(0);-moz-transform:scale(0);-webkit-transform:scale(0);-o-transform:scale(0);opacity:0;position:fixed;right:24px;bottom:140px;z-index:303;width:44px}
#s_popup_advert .advert-shrink2{bottom:184px}
#s_popup_advert .close-shrink{cursor:pointer;position:absolute;left:41px;top:-5px;color:#c0c2c8;font-size:12px}
#s_popup_advert .shrink-link{display:block;height:44px}
#s_popup_advert .shrink{display:block;width:100%;height:100%;border-radius:22px}
#s_popup_advert .replay{cursor:pointer;display:block;margin-top:6px;border-radius:4px;text-align:center;line-height:20px;font-size:13px;color:#9195a3}
#s_popup_advert .close-shrink:hover{color:#9195a3}
#s_popup_advert .replay:hover{color:#626675}
@media screen and (max-width:1158px){
#s_popup_advert{display:none}
}
.guide-info-new{z-index:999;height:34px;padding:0 15px;min-width:120px;background-color:rgba(98,102,117,.8);box-shadow:0 2px 10px 0 rgba(0,0,0,.1);border-radius:6px 6px 6px 6px;text-align:left;position:absolute;line-height:35px;white-space:nowrap}
.guide-info-new span{display:inline-block;vertical-align:top;font-size:13px;font-family:Arial,sans-serif;color:#fff;margin-right:-5px}
.guide-info-new .guide-close{color:#d7d9e0;margin-left:8px;display:inline-block;height:34px;text-align:center;vertical-align:top;margin-top:0;font-size:13px!important;cursor:pointer;transform:scale(.75)}
.guide-info-new .guide-arrow-left{right:-11px;top:10px;background:url('https://pss.bdstatic.com/static/superman/img/guide_new/arrow-left-a7b272965a.png')}
.guide-info-new .guide-arrow-top{bottom:-11px;left:10px;background:url('https://pss.bdstatic.com/static/superman/img/guide_new/arrow-top-d81f5f8843.png') no-repeat 0 0}
.guide-info-new .guide-arrow-right{left:-11px;top:10px;background:url('https://pss.bdstatic.com/static/superman/img/guide_new/arrow-right-69f7969669.png') no-repeat 0 0}
.guide-info-new .guide-arrow-left,.guide-info-new .guide-arrow-right,.guide-info-new .guide-arrow-top{position:absolute;opacity:.8;height:11px;width:11px;background-size:11px 11px}
.guide-info-new :hover .guide-close{color:#d7d9e0}
.red-point{position:relative}
.red-point::before{content:" ";border:3px solid #f73131;border-radius:3px;position:absolute;z-index:1000;right:0;margin-right:-5px;margin-top:-2px}
.color222{color:#222!important}
.guide-info-login{top:55px;width:250px;height:102px;right:29px;background:#f3f6ff;border-radius:8px;position:absolute;padding:3px 16px;box-sizing:border-box;box-shadow:none}
.guide-info-login .guide-left-main{float:left;position:absolute;left:12px;height:95px;padding:8px 0;box-sizing:border-box}
.guide-info-login .guide-left-main .guide-first-content,.guide-info-login .guide-left-main .guide-first-title{margin-top:2px;font-size:13px;color:#222;line-height:14px;width:98px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.guide-info-login .guide-left-main .guide-first-title{font-weight:bolder}
.guide-info-login .guide-left-main .guide-left-content{font-size:13px;color:#222;line-height:13px;margin:7px 0 0 0;width:98px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.guide-info-login .guide-right-main{float:right;width:120px;height:95px}
.guide-info-login .guide-close{position:absolute;right:6px;top:-4px;height:25px;z-index:3}
.guide-info-login .guide-close:hover{color:#315efb}
.guide-info-login .guide-arrow-bottom{width:0;height:0;position:absolute;top:-6px;right:18px;border-style:solid;border-width:0 4px 6px;box-sizing:border-box;border-color:transparent}
.guide-info-login .guide-left-button{background:#4e6ef2;border-radius:6px;width:80px;height:30px;color:#fff;text-align:center;line-height:30px;margin-top:13px;cursor:pointer;font-size:14px}
.guide-info-login .guide-left-button:hover{background:#315efb}
.guide-word-tips{position:absolute;border-radius:8px;box-sizing:border-box;right:-110px;top:51px;max-height:63px;line-height:21px;padding:6px 11px;font-size:13px;text-align:left}
.guide-word-tips a{color:#4e6ef2;cursor:pointer;text-decoration:none}
.guide-word-tips a:hover{color:#315efb;text-decoration:underline}
.guide-word-tips .word-guide-close{display:none}
.guide-word-tips .word-arrow-bottom-black,.guide-word-tips .word-arrow-bottom-white{position:absolute;width:0;height:0;top:-7px;right:12px;border-style:solid;border-width:0 5px 6px;box-sizing:border-box;border-color:transparent;border-bottom-color:#f2f2f2;z-index:1}
.guide-word-tips .word-arrow-bottom-black{right:11px}
.guide-word-tips .word-arrow-bottom-white{top:-6px;border-width:0 4px 6px;border-bottom-color:#fff;z-index:2}
.tip{max-width:206px;color:#fff;background:#626675;padding-right:11px;border:1px solid #626675}
.tip .word-arrow-bottom-black{right:11px;top:-7px;border-bottom-color:#626675;border-width:0 4px 6px}
.tip .word-arrow-bottom-white{display:none}
.operate{max-width:216px;color:#222;background:#fff;border:1px solid rgba(0,0,0,.05);box-shadow:0 2px 4px 0 rgba(0,0,0,.1);padding-right:24px}
.operate .word-guide-close{display:block;position:absolute;right:6px;top:6px;z-index:3;color:#c4c7ce;font-size:12px;transform:scale(.75);cursor:pointer}
.operate .word-guide-close:hover{color:#315efb}
.guide-info-pos-0{right:35px}
.guide-info-pos-2{right:104px}
.is-login.guide-info-pos-1{right:93px}
.is-login.guide-info-pos-2{right:145px}
.guide-info-login.guide-info-pos-0{right:26px}
.guide-info-login.guide-info-pos-2{right:95px}
.has-background .darkmode.blue,.has-background .darkmode.dark{background-color:none}
.s-manhattan-index .aging-scale #bottom_layer{position:relative}
.wrapper-nologin-more-screen.aging-scale #bottom_layer{position:relative}
.s-p-top .index-logo-aging-tools{display:none}
.darkmode.dark #s_main{background:rgba(31,31,37,.85)}
.darkmode.dark #bottom_layer,.darkmode.dark #head,.darkmode.dark #s_menu_gurd.s-down,.darkmode.dark #s_top_wrap,.darkmode.dark #s_top_wrap.s-down .s-center-box,.darkmode.dark #s_wrap{background-color:#1f1f25}
.darkmode.blue #s_main{background:rgba(20,30,66,.85)}
.darkmode.blue #bottom_layer,.darkmode.blue #head,.darkmode.blue #s_menu_gurd.s-down,.darkmode.blue #s_top_wrap,.darkmode.blue #s_top_wrap.s-down .s-center-box,.darkmode.blue #s_wrap{background-color:#141e42}
.darkmode .s-bottom-space{margin-top:0}
.darkmode.blue .ai-dark,.darkmode.blue .ai-dark-title,.darkmode.blue .ai-tool-title,.darkmode.blue .mnav.s-top-more-btn .s-bri,.darkmode.blue .s-ctner-menus .current,.darkmode.blue .s-ctner-menus .menus-wrapper,.darkmode.blue .s-hotsearch-wrapper .hot-refresh,.darkmode.blue .s-hotsearch-wrapper .s-hotsearch-title .title-text,.darkmode.blue .s-news-rank-wrapper .s-rank-title div,.darkmode.blue .s-top-left .mnav,.darkmode.blue .s-top-right .s-top-right-text,.darkmode.blue .s-top-right .s-top-username .s-top-img-wrapper,.darkmode.blue .s-top-right .s-top-username .user-name,.darkmode.blue .s-weather-wrapper .s-mod-weather .unknown-city .unknown-setting,.darkmode.blue .s-weather-wrapper .show-city-name,.darkmode.blue .s-weather-wrapper .show-icon-temp,.darkmode.dark .ai-dark,.darkmode.dark .ai-dark-title,.darkmode.dark .ai-tool-title,.darkmode.dark .mnav.s-top-more-btn .s-bri,.darkmode.dark .s-ctner-menus .current,.darkmode.dark .s-ctner-menus .menus-wrapper,.darkmode.dark .s-hotsearch-wrapper .hot-refresh,.darkmode.dark .s-hotsearch-wrapper .s-hotsearch-title .title-text,.darkmode.dark .s-news-rank-wrapper .s-rank-title div,.darkmode.dark .s-top-left .mnav,.darkmode.dark .s-top-right .s-top-right-text,.darkmode.dark .s-top-right .s-top-username .s-top-img-wrapper,.darkmode.dark .s-top-right .s-top-username .user-name,.darkmode.dark .s-weather-wrapper .s-mod-weather .unknown-city .unknown-setting,.darkmode.dark .s-weather-wrapper .show-city-name,.darkmode.dark .s-weather-wrapper .show-icon-temp{color:#ffd862!important}
.darkmode.blue .ai-dark:hover,.darkmode.dark .ai-dark:hover{color:#315efb}
.darkmode.blue .ai-dark:hover,.darkmode.blue .ai-title-left-link .ai-title-left-link-text:hover,.darkmode.blue .ai-title-left-link .c-icon:hover,.darkmode.blue .mnav.s-top-more-btn .s-bri:hover,.darkmode.blue .s-ctner-menus .current:hover,.darkmode.blue .s-ctner-menus .menus-wrapper:hover,.darkmode.blue .s-ctner-menus .s-menu-item:hover,.darkmode.blue .s-hotsearch-wrapper .hot-refresh:hover,.darkmode.blue .s-hotsearch-wrapper .s-hotsearch-title .title-text:hover,.darkmode.blue .s-news-rank-wrapper .s-rank-title .hot-refresh:hover,.darkmode.blue .s-news-rank-wrapper .s-rank-title div:hover,.darkmode.blue .s-top-left .mnav:hover,.darkmode.blue .s-top-right .s-top-right-text:hover,.darkmode.blue .s-top-right .s-top-username .user-name:hover,.darkmode.blue .s-weather-wrapper .s-mod-weather .unknown-city .unknown-setting:hover,.darkmode.blue .s-weather-wrapper .show-city-name:hover,.darkmode.blue .s-weather-wrapper .show-icon-temp:hover,.darkmode.dark .ai-dark:hover,.darkmode.dark .ai-title-left-link .ai-title-left-link-text:hover,.darkmode.dark .ai-title-left-link .c-icon:hover,.darkmode.dark .mnav.s-top-more-btn .s-bri:hover,.darkmode.dark .s-ctner-menus .current:hover,.darkmode.dark .s-ctner-menus .menus-wrapper:hover,.darkmode.dark .s-ctner-menus .s-menu-item:hover,.darkmode.dark .s-hotsearch-wrapper .hot-refresh:hover,.darkmode.dark .s-hotsearch-wrapper .s-hotsearch-title .title-text:hover,.darkmode.dark .s-news-rank-wrapper .s-rank-title .hot-refresh:hover,.darkmode.dark .s-news-rank-wrapper .s-rank-title div:hover,.darkmode.dark .s-top-left .mnav:hover,.darkmode.dark .s-top-right .s-top-right-text:hover,.darkmode.dark .s-top-right .s-top-username .user-name:hover,.darkmode.dark .s-weather-wrapper .s-mod-weather .unknown-city .unknown-setting:hover,.darkmode.dark .s-weather-wrapper .show-city-name:hover,.darkmode.dark .s-weather-wrapper .show-icon-temp:hover{color:#fff762!important}
.darkmode.blue .ai-dark-title:hover,.darkmode.dark .ai-dark-title:hover{color:#315efb!important}
.darkmode.blue .s-ctner-menus .s-menu-item-underline,.darkmode.dark .s-ctner-menus .s-menu-item-underline{background:#a8acad}
.darkmode #bottom_layer a,.darkmode #s_wrap .s-setbar,.darkmode #u1 .s-top-right .s-top-right-text.c-color-t,.darkmode .s-block-nav .d-nav-item .border-for-item,.darkmode .s-block-stock .stock-item .stock-code,.darkmode .s-block-stock .stock-item .stock-code a,.darkmode .s-block-stock .stock-item .stock-title,.darkmode .s-block-stock .stock-item .stock-title a,.darkmode .s-code-blocks .nav-item .border-for-item,.darkmode .s-code-blocks .s-setbar .set-btn,.darkmode .s-code-blocks .s-setbar .set-refresh .c-icon{color:#ffd862}
.darkmode #bottom_layer a:hover,.darkmode #s_wrap .border-for-item:hover .nav-text,.darkmode #s_wrap .s-setbar:hover,.darkmode #u1 .s-top-right .s-top-right-text.c-color-t:hover,.darkmode .s-block-stock .stock-item .stock-code a:hover,.darkmode .s-block-stock .stock-item .stock-code:hover,.darkmode .s-block-stock .stock-item .stock-title a:hover,.darkmode .s-block-stock .stock-item .stock-title:hover,.darkmode .s-code-blocks .s-setbar .set-btn:hover,.darkmode .s-code-blocks .s-setbar .set-refresh .c-icon:hover{color:#fff762}
.darkmode .ai-dark-subtitle,.darkmode .s-block-stock .stock-item .stock-highprice,.darkmode .s-block-stock .stock-item .stock-lowprice,.darkmode .s-block-stock .stock-item .stock-updatetime,.darkmode .s-skin-hasbg .s-block-stock .stock-banner,.darkmode .s-xmancard-mine_new .tips-manager-btn .manager-btn-text{color:#a8acad}
.darkmode #result_logo .index-logo-peak{display:block}
.darkmode #result_logo .index-logo-src,.darkmode #result_logo .index-logo-srcnew{display:none}
.darkmode .s-p-top .index-logo-aging-tools{display:block}
.darkmode .s-p-top .index-logo-src,.darkmode .s-p-top .index-logo-srcnew,.darkmode .s-p-top .s_lg_img_gold_show,.darkmode .s-p-top .s_lg_img_gold_showre{display:none!important}
.darkmode #s-user-name-menu a,.darkmode #s-user-name-menu a:visited,.darkmode #s-user-setting-menu a,.darkmode #s-user-setting-menu a:visited{color:#333}
.darkmode #s-user-name-menu a:hover,.darkmode #s-user-setting-menu a:hover{color:#315efb}
body.big-event-gray #aging-tools-pc div[class^=item-wrap],body.big-event-gray #bottom_layer,body.big-event-gray #bottom_layer .tip-hover-panel,body.big-event-gray #form,body.big-event-gray #passport-login-pop,body.big-event-gray #s-hotsearch-wrapper,body.big-event-gray #s_content_1,body.big-event-gray #s_content_100,body.big-event-gray #s_menu_gurd,body.big-event-gray #s_side_wrapper,body.big-event-gray #u1,body.big-event-gray .advert-shrink,body.big-event-gray .bdlayer,body.big-event-gray .popup-advert,body.big-event-gray .s-ctner-menus .s-menu-item-underline,body.big-event-gray .s-menu-container,body.big-event-gray .s-news-rank-wrapper,body.big-event-gray .s-skin-container.skin-gray-event,body.big-event-gray .s-top-left-new.s-isindex-wrap,body.big-event-gray .under-searchbox-tips,body.big-event-gray div[class^=cards_pop]{filter:progid:DXImageTransform.Microsoft.BasicImage(grayscale=1);-webkit-filter:grayscale(100%);-moz-filter:grayscale(100%);-ms-filter:grayscale(100%);-o-filter:grayscale(100%);filter:grayscale(100%);filter:gray}
body.big-event-gray #head_wrapper .sam_search .s_btn,body.big-event-gray #su,body.big-event-gray .wrapper_new .s_btn_wr .s_btn{background-color:#222}
body.big-event-gray .c-icon.title-content-top-icon,body.big-event-gray .title-content-index.c-index-single.c-index-single-hot1,body.big-event-gray .title-content-index.c-index-single.c-index-single-hot2,body.big-event-gray .title-content-index.c-index-single.c-index-single-hot3{color:#626675}
body.big-event-gray #head_wrapper .sam_search #kw{border-color:#222}
body.big-event-gray #s_content_2 .water-container{width:670px}
body.big-event-gray .hot-news-wrapper .s-news-rank-content .title-content:hover,body.big-event-gray .s-hotsearch-wrapper .s-hotsearch-content .title-content:hover{text-decoration:underline;color:#222}
body.big-event-gray .hot-news-wrapper .s-rank-title .hot-refresh:hover,body.big-event-gray .hot-news-wrapper .s-rank-title .hot-refresh:hover .hot-refresh-text,body.big-event-gray .hot-news-wrapper .s-rank-title .hot-refresh:hover .refresh-icon,body.big-event-gray .hot-news-wrapper .s-rank-title .title-text:hover,body.big-event-gray .s-hotsearch-wrapper .s-hotsearch-title .hot-refresh:hover,body.big-event-gray .s-hotsearch-wrapper .s-hotsearch-title .hot-refresh:hover .c-icon,body.big-event-gray .s-hotsearch-wrapper .s-hotsearch-title .hot-refresh:hover .hot-refresh-text,body.big-event-gray .s-hotsearch-wrapper .s-hotsearch-title .title-text:hover{text-decoration:none;color:#222}
.sui-draggable-mask{position:fixed;width:100%;height:100%;z-index:2001;left:0;top:0;user-select:none;-moz-user-select:none;-webkit-user-select:none;-ms-user-select:none}
.sui-draggable{opacity:.5;z-index:7}
.sui-draggable .drag-mask{border:1px dashed #38f}
.sui-draggable,.sui-draggable-wraper{z-index:2000}
.sui-componentWrap{zoom:1}
.sui-wraper{text-align:left}
.sui-draggable-proxy{visibility:hidden}
.sui-draggsort-collapse{visibility:hidden;width:100%}
.sui-draggsort-holder{border:0!important;position:absolute}
.new_search_guide_bub{margin-top:35px;font-size:16px;line-height:16px;display:flex;justify-content:center}
.new_search_guide_bub_container{display:flex;align-items:center;padding:8px;background:linear-gradient(0deg,rgba(51,119,255,.05),rgba(51,119,255,.05)),linear-gradient(90deg,rgba(229,231,255,.06) -4.77%,rgba(212,222,255,.3) 88.06%);background-color:#fff;border-radius:29px}
.new_search_guide_bub_container:hover{text-decoration:none}
.new_search_guide_bub_container:active,.new_search_guide_bub_container:visited{text-decoration:none;color:unset}
.new_search_guide_bub_container_left{margin-right:6px;position:relative;margin-left:3px;margin-bottom:-1px}
.new_search_guide_bub_container_left .search_icon{width:16px;height:16px}
.new_search_guide_bub_container_left .four_point_star{position:absolute;top:-1px;right:0}
.new_search_guide_bub_container_center{font-size:16px;font-weight:400;line-height:25px;height:24px;letter-spacing:.06em;text-align:left;background:#000;color:transparent;background-clip:text;-webkit-background-clip:text;display:flex}
.new_search_guide_bub_container_center .prefix_text{margin-left:5px}
.new_search_guide_bub_container_center_number_counter{display:flex;overflow:hidden}
.new_search_guide_bub_container_center_number_counter_digit_container_digit_list{display:flex;flex-direction:column;transition:transform var(--duration) var(--easing)}
.new_search_guide_bub_container_center_number_counter_digit_container_digit_list .digit{height:20px}
.new_search_guide_bub_container_right{display:flex;justify-content:center;align-items:center;width:24px;height:24px;font-size:12px;margin-left:6px;border-radius:50%;background:linear-gradient(90deg,rgba(62,75,255,.1) 0,rgba(0,13,255,.1) 100%);color:#000}
.new_search_guide_bub.new_search_guide_bub_animation .new_search_guide_bub_container_left{animation:new_search_search_icon_rotate 1s ease-in-out;transform-origin:bottom right}
.new_search_guide_bub.new_search_guide_bub_animation .new_search_guide_bub_container_left .four_point_star{animation:new_search_four_point_star_rotate 1.42s ease-in-out;transform-origin:center center}
.new_search_guide_bub.new_search_guide_bub_animation .new_search_guide_bub_container_center{background-image:linear-gradient(90deg,#ff1E0033 0,#ff1e00 20%,#335bff 80%,#335BFF33 100%);background-size:120% 100%;background-position:var(--bubble-animation-length) 0;background-repeat:no-repeat;animation:new_search_guide_bub_shine 1s linear;animation-delay:.5s}
.new_search_guide_bub.new_search_guide_bub_animation .new_search_guide_bub_container_right .cos-icon{animation:new_search_arrow_right_icon_rotate 1s linear;animation-delay:1.2s}
#head_wrapper.s-down .new_search_guide_bub{display:none}
#head_wrapper.s-upgrade-down .new_search_guide_bub{display:none}
#head_wrapper.s-ps-islite .new_search_guide_bub{position:absolute;bottom:-35px;left:0;width:100%;transform:translateY(100%)}
@keyframes new_search_four_point_star_rotate{
0%{transform:rotate(0)}
12%{transform:rotate(-20deg)}
36%{transform:rotate(180deg)}
100%{transform:rotate(0)}
}
@keyframes new_search_search_icon_rotate{
0%{transform:rotate(0)}
10%{transform:rotate(-10deg)}
40%{transform:rotate(25deg)}
100%{transform:rotate(0)}
}
@keyframes new_search_arrow_right_icon_rotate{
0%{transform:translateX(0)}
16%{transform:translateX(2px)}
26%{transform:translateX(0)}
36%{transform:translateX(-2px)}
52%{transform:translateX(0)}
68%{transform:translateX(2px)}
82%{transform:translateX(0)}
100%{transform:translateX(0)}
}
@keyframes new_search_guide_bub_shine{
0%{background-position:-150px 0}
100%{background-position:150px 0}
}</style><style type="text/css" index="weather">.s-weather-wrapper{float:none;display:inline-block;margin-left:32px;margin-top:19px;height:23px;vertical-align:top}
.s-weather-wrapper .s-mod-weather{margin:0;height:23px;line-height:23px;display:inline-block}
.s-weather-wrapper .s-mod-weather .weather-mod{display:none}
.s-weather-wrapper .s-mod-weather .weather-mod .city-wather{display:inline-block}
.s-weather-wrapper .s-mod-weather .weather-mod .city-wather .show-city{height:23px;display:inline-block;vertical-align:top;margin-right:8px}
.s-weather-wrapper .s-mod-weather .weather-mod .city-wather .show-icon{margin-right:8px}
.s-weather-wrapper .s-mod-weather .weather-mod .city-wather .show-icon .weather-icon{width:23px;height:23px;display:inline-block;background-size:100% 100%;margin-right:4px}
.s-weather-wrapper .s-mod-weather .weather-mod .city-wather .show-icon .show-icon-temp{vertical-align:top}
.s-weather-wrapper .s-mod-weather .weather-mod .show-pollution{display:inline-block;vertical-align:top;height:23px;line-height:23px}
.s-weather-wrapper .s-mod-weather .weather-mod .show-pollution .show-polution-name{height:18px;color:#fff;font-size:11px;display:inline-block;line-height:17px;text-align:center;padding:0 6px}
.s-weather-wrapper .s-mod-weather .weather-mod .show-pollution .show-airParm{background-color:#cacaca;border-radius:9px}
.s-weather-wrapper .s-mod-weather .weather-mod .show-pollution .polution-level-0{background-color:#40c057}
.s-weather-wrapper .s-mod-weather .weather-mod .show-pollution .polution-level-10{background-color:#82c91e}
.s-weather-wrapper .s-mod-weather .weather-mod .show-pollution .polution-level-20{background-color:#f76707}
.s-weather-wrapper .s-mod-weather .weather-mod .show-pollution .polution-level-30{background-color:#e03131}
.s-weather-wrapper .s-mod-weather .weather-mod .show-pollution .polution-level-40{background-color:#841c3c}
.s-weather-wrapper .s-mod-weather .weather-mod .show-pollution .polution-level-50{background-color:#540822}
.s-weather-wrapper .s-mod-weather .unknown-city .unknown-icon{color:#4e71f2;font-size:20px;background:0 0;margin-right:4px}
.s-weather-wrapper .s-mod-weather .unknown-city .unknown-setting{vertical-align:top;color:#222}
.s-weather-wrapper .s-mod-weather.hide-unknow-city .weather-mod{display:block}
.s-weather-wrapper .s-mod-weather.hide-unknow-city .unknown-city{display:none}
.s-weather-wrapper:hover a{text-decoration:none}
.s-weather-wrapper:hover .show-city-name,.s-weather-wrapper:hover .show-icon-temp{color:#315efb}
.s-weather-wrapper:hover .unknown-city .unknown-setting{color:#315efb}
.s-skin-hasbg .s-weather-wrapper .show-city-name,.s-skin-hasbg .s-weather-wrapper .show-icon-temp{color:rgba(255,255,255,.85)}
.s-skin-hasbg .s-weather-wrapper .unknown-city .unknown-setting{color:rgba(255,255,255,.85)}
.s-skin-hasbg .s-weather-wrapper:hover .show-city-name,.s-skin-hasbg .s-weather-wrapper:hover .show-icon-temp{color:#fff}
.s-skin-hasbg .s-weather-wrapper:hover .unknown-city .unknown-setting{color:#fff}
.s-mod-setweather{position:absolute;top:48px;right:24px;width:496px;z-index:500;display:none;background-color:#fff;box-shadow:0 2px 10px 0 rgba(0,0,0,.1);border-radius:12px;text-align:left}
.s-mod-setweather span{user-select:none}
.s-mod-setweather.hide{display:none}
.s-mod-setweather.show{display:block}
.s-mod-setweather.show-setting .setweather-content,.s-mod-setweather.show-setting .warn-icon-link{display:none}
.s-mod-setweather.hide-warning .warn-icon-link,.s-mod-setweather.hide-warning .warn-mod{display:none}
.s-mod-setweather.show-content .setweather-setting{display:none}
.s-mod-setweather .warn-icon-link{position:absolute;top:21px;left:22px;width:24px;height:24px;background-image:url('https://pss.bdstatic.com/static/superman/img/weather/warn@2x-e89baf05e1.png');background-size:100% 100%;_background-image:none;_filter:progid:DXImageTransform.Microsoft.AlphaImageLoader(src='https://pss.bdstatic.com/static/superman/img/weather/warn@1x-789770467b.png', enabled=true, sizingMethod="crop")}
.s-mod-setweather .setweather-content{padding:12px 16px 16px}
.s-mod-setweather .setweather-content .warn-mod{height:36px;background:#f5f5f6;border-radius:6px;margin-top:4px;margin-bottom:12px}
.s-mod-setweather .setweather-content .warn-mod .warn-img-bg{float:left;width:36px;height:36px;border-radius:6px}
.s-mod-setweather .setweather-content .warn-mod .warn-title{float:left;margin-top:6px;margin-left:10px}
.s-mod-setweather .setweather-content .warn-mod .warn-title:hover{text-decoration:none}
.s-mod-setweather .setweather-content .lunar-mod .lunar-festival{margin-left:16px}
.s-mod-setweather .setweather-content .lunar-mod .lunar-sevenday{margin-left:16px;color:#315efb}
.s-mod-setweather .setweather-content .lunar-mod .lunar-setting-btn{float:right;cursor:pointer}
.s-mod-setweather .setweather-content .lunar-mod .lunar-setting-btn .c-icon{color:#4e71f2;display:inline-block;margin-right:4px;font-size:18px;margin-top:3px;line-height:17px;vertical-align:top}
.s-mod-setweather .setweather-content .lunar-mod .lunar-setting-btn:hover .lunar-settint-text{color:#315efb}
.s-mod-setweather .setweather-content .everyday-mod{text-align:center}
.s-mod-setweather .setweather-content .everyday-mod .everyday-item{float:left;width:84px;overflow:hidden;margin-right:11px;margin-top:13px;padding:5px 0}
.s-mod-setweather .setweather-content .everyday-mod .everyday-item:hover{background:#f5f5f6;border-radius:6px}
.s-mod-setweather .setweather-content .everyday-mod .everyday-item:last-child{margin-right:0}
.s-mod-setweather .setweather-content .everyday-mod .everyday-icon{width:32px;height:32px;margin:3px auto;display:block}
.s-mod-setweather .setweather-content .everyday-mod .everyday-condition,.s-mod-setweather .setweather-content .everyday-mod .everyday-temp,.s-mod-setweather .setweather-content .everyday-mod .everyday-wind{font-size:12px;line-height:22px}
.s-mod-setweather .setweather-content .everyday-mod .everyday-condition,.s-mod-setweather .setweather-content .everyday-mod .everyday-wind{margin-top:-2px}
.s-mod-setweather .setweather-setting{padding:12px 16px 16px}
.s-mod-setweather .setweather-setting .c-select{width:122px}
.s-mod-setweather .setweather-setting .c-select-dropdown{width:120px}
.s-mod-setweather .setweather-setting .setting-city-title{text-align:left}
.s-mod-setweather .setweather-setting .setting-city-select{margin-top:11px;margin-bottom:12px;text-align:left}
.s-mod-setweather .setweather-setting .setting-city-select .setting-city-province{margin-right:16px}
.s-mod-setweather .setweather-setting .setting-extreme-weather{text-align:left}
.s-mod-setweather .setweather-setting .setting-extreme-weather .setting-extreme-weather-checkbox{margin-bottom:4px}
.s-mod-setweather .setweather-setting .setting-extreme-weather .setting-extreme-weather-label{display:inline-block;padding-left:8px}
.s-mod-setweather .setweather-setting .setting-save-or-cancel{margin-top:15px;text-align:left}
.s-mod-setweather .setweather-setting .setting-save-or-cancel .setting-save{margin-left:16px}
.s-mod-setweather .setweather-setting .setting-save-or-cancel .setting-cancel{cursor:pointer}</style><style type="text/css" index="super_min">html{height:100%}
body{height:100%;min-width:1250px;width:100%;cursor:default}
#head{position:relative;height:100%;width:100%;min-height:768px;cursor:default}
.is-lite #head{min-height:600px}
#s_wrap{width:100%;_width:1000px;margin:0 auto;position:relative}
#u_sp .s-user-name-top{position:relative}
#u_sp .s-user-name-top a{text-decoration:none;height:24px;line-height:24px;padding-left:7px;width:72px;-moz-user-select:none;cursor:pointer;text-align:left;display:block;color:#0079f5;margin:0}
#u_sp .s-user-name-top a:hover{background:#f1f8ff;text-decoration:none}
#u_sp .s-user-name-top a.sep{border-top:1px solid #ebebeb}
.s-user-set-menu{padding-top:6px;top:26px;position:absolute;right:75px;*right:73px;float:right;width:100px;z-index:999;text-align:left}
.s-user-set-menu div{position:relative;border:1px solid #e3e3e3;box-shadow:1px 1px 5px #d1d1d1;-moz-box-shadow:1px 1px 5px #d1d1d1;-webkit-box-shadow:1px 1px 5px #d1d1d1}
.s-user-set-menu div a{-ms-filter:""}
.s-user-set-menu.menu-top{width:70px}
.s-user-set-menu .menu-arrow{position:absolute;z-index:2;top:-4px;left:28px;display:inline-block;width:0;height:0;line-height:0;border:5px dashed transparent;border-bottom:5px solid #e3e3e3;font-size:0}
.s-user-set-menu .menu-arrow em{position:absolute;top:-4px;left:-5px;display:block;width:0;height:0;line-height:0;font-size:0;color:#666;border-top:5px dashed transparent;border-left:5px dashed transparent;border-right:5px dashed transparent;border-bottom:5px solid #fff}
.s-user-set-menu.menu-set .menu-arrow{float:left;left:27px}
.s-user-set-menu.menu-top .menu-arrow{float:right;right:32px}
#s_user_name_menu.s-user-set-menu a,#s_user_setting_menu.s-user-set-menu a{text-shadow:none;color:#000}
#s_user_name_menu.s-user-set-menu a:hover,#s_user_setting_menu.s-user-set-menu a:hover{background:#38f;color:#fff;text-decoration:none}
.s-user-set-menu a{text-decoration:none;height:25px;line-height:25px;padding-left:8px;background-color:#fff;-moz-user-select:none;cursor:pointer;text-align:left;display:block;color:#333;margin:0;font-size:12px}
.s-user-set-menu a.sep{border-top:1px solid #ebebeb}
.s-bottom-ctner{width:100%;position:absolute;z-index:3;bottom:12px;left:0;overflow:hidden;color:#999;zoom:1;margin:0}
.is-lite #bottom_container .qr-style{overflow:hidden;width:640px;margin:0 auto;text-align:left}
.is-lite #bottom_container .qr-style{margin-bottom:68px}
#bottom_container .qrcode-layer{display:none}
#bottom_container .no-qrcode-layer .lh{display:inline}
.is-lite #bottom_container .qr-style .qrcode-layer{display:block}
.is-lite #bottom_container .qr-style .no-qrcode-layer .lh{display:block;text-align:left;margin:12px 0 6px}
.is-lite #bottom_container .qr-style .no-qrcode-layer,.is-lite #bottom_container .qr-style .qrcode-layer,.is-lite #bottom_container .qr-style .qrcode-layer .qrcode-img,.is-lite #bottom_container .qr-style .qrcode-layer .qrcode-item,.is-lite #bottom_container .qr-style .qrcode-layer .qrcode-text{display:inline-block;*display:inline;zoom:1;vertical-align:top}
.is-lite #bottom_container .qr-style .no-qrcode-layer{margin-left:33px}
.is-lite #bottom_container .qrcode-layer .qrcode-img{height:60px;width:60px;background:url(../../mancard/img/qrcode.png) no-repeat}
.is-lite #bottom_container .qrcode-layer .qrcode-item-2 .qrcode-img{background-position:0 -61px}
#bottom_container .qrcode-layer .qrcode-text{color:#999;line-height:23px;margin:8px 0 0 10px}
#bottom_container .qrcode-layer .qrcode-text b{color:#4c4c4c;font-weight:700}
.s-skin-hasbg #bottom_container .qrcode-layer .qrcode-text,.s-skin-hasbg #bottom_container .qrcode-layer .qrcode-text b{color:#fff}
.is-lite #bottom_container .qrcode-layer .qrcode-item-2{margin-left:33px}
#s_wrap .s-bottom-ctner{position:static;margin:23px auto 12px}
.s-bottom-ctner .sethome{margin-right:12px}
.s-bottom-ctner a{font-size:12px;text-decoration:underline;color:#999}
.s-bottom-ctner a:hover{text-decoration:underline}
#f_bottom_container.first-screen{position:fixed;bottom:0;left:0;z-index:301;height:36px;line-height:36px;padding:0;box-shadow:0 -2px 2px #eee;background-image:-webkit-gradient(linear,left top,left bottom,from(rgba(255,255,255,.9)),to(rgba(255,255,255,.9)));background-image:-moz-linear-gradient(rgba(255,255,255,.9) 0,rgba(255,255,255,.9) 100%);background-image:-ms-linear-gradient(rgba(255,255,255,.9) 0,rgba(255,255,255,.9) 100%);background-image:-o-linear-gradient(rgba(255,255,255,.9) 0,rgba(255,255,255,.9) 100%);background-image:linear-gradient(rgba(255,255,255,.9) 0,rgba(255,255,255,.9) 100%);filter:progid:DXImageTransform.Microsoft.gradient(startColorstr='#e5ffffff', endColorstr='#e5ffffff')}
.s-bottom-copyright{font-size:15px;height:20px;width:40px;position:relative;bottom:-2px}
.s-bottom-ctner img{position:relative;margin:0 0 -2px 0}
.recordcode{padding-top:6px;height:16px;line-height:16px}
.recordcode a{text-decoration:none}
.recordcode a i{display:inline-block;width:14px;height:16px;margin-right:6px;background:url(../img/icon-police-7296bdfd98.png) 0 0 no-repeat;vertical-align:middle;margin-top:-4px}
.s-add-bar{position:absolute;height:318px;width:35px;left:0;top:250px}
.s-add-bar .s-add-bar-bg{height:318px;width:35px;background-color:#eee}
.s-add-bar a{display:inline-block;height:318px;width:35px;position:absolute;top:0;left:0;background:url(../img/s_add_bar-4de4b69586.png) no-repeat 0 110px}
.s-bd-fm-search-tip{width:185px;height:46px;position:absolute;left:666px;top:0;color:#fff}
.s-bd-fm-search-tip .bg-img{position:absolute;left:0;top:0;width:185px;height:46px;background:url(../img/music/music_tip-c8500eb8e4.png) no-repeat 0 0;_background:0 0;_filter:progid:DXImageTransform.Microsoft.AlphaImageLoader(enabled=true, sizingMethod=scale, src="http://su.bdimg.com/static/superman/img/music/music_tip-c8500eb8e4.png")}
.s-bd-fm-search-tip .content{position:absolute;left:0;width:168px;top:8px;height:32px;line-height:16px;margin-left:17px}
.s-bd-fm-search-tip .content .first-line{display:block;text-indent:15px;background:url(../img/music/music_playing-ebd7a54fa2.gif) no-repeat}
.s-bd-fm-search-tip .close-btn{width:15px;height:15px;position:absolute;right:2px;top:2px;cursor:pointer}
.s-xmancard{min-height:318px;_height:318px}
.tang-pass-pop-accRealName .pass-realName-loginStitle{text-align:left}
#wrapper .s-top-right.s-down.ipad-min-width-down,#wrapper.ipad-min-width,body .s-top-right.s-down.ipad-min-width-down,body.ipad-min-width{min-width:1024px}
.s-skin-hasbg{background:0 0;_zoom:1}
.s-skin-container{position:fixed;_position:absolute;top:0;left:0;height:100%;width:100%;min-width:1000px;z-index:-10;background-position:center 0;background-repeat:no-repeat;background-size:cover;-webkit-background-size:cover;-o-background-size:cover;zoom:1}
@media screen and (max-width:1250px){
.s-skin-container{min-width:1250px}
}
.s-skin-hasbg.red-blue-logo #lm,.s-skin-hasbg.red-blue-logo #lm a,.s-skin-hasbg.red-blue-logo #nv a{color:#333}
.s-skin-hasbg.red-blue-logo #lm a{color:#fff}
.s-skin-hasbg.red-blue-logo .s-ps-islite #lm a{color:#333}
.s-skin-hasbg.red-blue-logo #nv b{color:#000}
.s-skin-hasbg.white-logo #lm,.s-skin-hasbg.white-logo #lm a,.s-skin-hasbg.white-logo #nv a,.s-skin-hasbg.white-logo #nv b{color:#fff}
.s-skin-hasbg #lm,.s-skin-hasbg #lm a,.s-skin-hasbg #nv a,.s-skin-hasbg #nv b,.s-skin-hasbg .s-sp-menu a{color:#fff}
.s-skin-hasbg .s-sp-menu a{_color:#fff}
.s-skin-hasbg .s-u-level{color:#fff}
.s-skin-hasbg #u_sp .s-icon{background:url(../img/skin_dark-48fa9dbf5c.png) no-repeat;_background:url(../img/skin_dark_ie-8647a11021.png) no-repeat}
.s-skin-hasbg #u_sp .s-icons .s-icon-treasure{background-position:-13px -577px;_background-position:-13px -577px}
.s-skin-hasbg #u_sp .s-icons .s-icon-lite{background-position:-280px -577px;_background-position:-280px -577px}
.s-skin-hasbg #u_sp .s-icons .s-icon-skin{background-position:-43px -578px;_background-position:-43px -578px}
.s-skin-hasbg #u_sp .s-icons .s-icon-msg{background-position:-228px -575px;_background-position:-238px -575px}
.s-skin-hasbg #u_sp .s-icons .s-icon-feedback{background-position:-72px -577px;_background-position:-72px -577px}
.s-skin-hasbg #u_sp .s-icons .s-icon-line{background-position:-104px -577px;_background-position:-104px -577px}
.s-skin-hasbg .s-icons .s-icon-feedback:hover{background-position:-196px -577px;_background-position:-197px -578px}
.s-skin-hasbg .s-user-name-top .user-arrow,.s-skin-hasbg .s-user-setting-top .user-arrow{background:url(../img/skin_dark-48fa9dbf5c.png) no-repeat -707px -588px;_background:url(../img/skin_dark_ie-8647a11021.png) no-repeat -707px -588px}
.s-skin-hasbg .s-user-name-top a{color:#0079f5}
.s-skin-hasbg .s-user-set-menu div{border-color:#fff}
.s-skin-hasbg .s-user-set-menu .menu-arrow{background-position:0 -90px}
.s-skin-hasbg .s-bottom-ctner,.s-skin-hasbg .s-bottom-ctner a{color:#fff}
.s-skin-hasbg .first-screen .sethome{color:#0079f5}
.s-skin-hasbg .first-screen,.s-skin-hasbg .first-screen a{font-size:12px;text-decoration:none;color:#999}
.s-skin-user #s_bottom_ctner{_bottom:-1px}
.s-skin-hasbg .bottom{background:0 0;border-top:none;padding-bottom:20px}
.s-skin-hasbg #s_bottom_ctner .bottom-line1,.s-skin-hasbg #s_bottom_ctner .bottom-line1 a,.s-skin-hasbg #s_seth a{color:#fff}
.s-skin-hasbg #s_bottom_ctner .bottom-line2,.s-skin-hasbg #s_bottom_ctner .bottom-line2 a{color:#ccc}
.s-skin-hasbg #mCon{background-color:transparent}
.s-skin-hasbg #mCon span{color:#fff}
.s-skin-hasbg #mMenu{background:rgba(0,0,0,.7);background:#000\9;filter:alpha(opacity=70)\9}
.s-skin-hasbg #mMenu a:hover{background:rgba(34,34,34,.7);background:#222}
.s-skin-hasbg #mMenu a,.s-skin-hasbg #mMenu a:visited{color:#fff}
.s-skin-hasbg .weather-setting-arraw{background:url(../img/load_ext_spis-5e9f273bb5.png) no-repeat 0 -90px}
.s-skin-hasbg .city-wather .show-vertical,.s-skin-hasbg .city-wather a,.s-skin-hasbg .city-wather em,.s-skin-hasbg .s-mod-weather .unknown-text,.s-skin-hasbg .show-pollution a,.s-skin-hasbg .show-pollution em,.s-skin-hasbg .show-pollution span,.s-skin-hasbg .unknown-city{color:#fff}
.s-skin-hasbg .city-wather em{color:#fff}
.s-skin-hasbg .show-pollution .show-polution-name{text-shadow:none}
.s-skin-hasbg .s-mod-weather .unknown-icon{background:url(../img/load_spis-64604bbf24.png) no-repeat -15px 0;_background:0 0;_filter:progid:DXImageTransform.Microsoft.AlphaImageLoader(src="http://su.bdimg.com/static/superman/img/ie6/weather_nobg_unknown_city-6a31f9a7a2.png", enabled=true, sizingMethod="crop")}
.s-skin-hasbg .s-mod-weather .unknown-text{color:#fff}
.s-skin-hasbg .s-mod-weather .unknown-setting{color:#8bc5ff}
.s-skin-hasbg .weather-setting-content{border:none}
.s-show-skin .weather-setting-content{border:1px solid #e3e3e3}
.s-skin-hasbg .s-ctner .s-ctner-menus{background:0 0;border-right:none;*z-index:10}
.s-skin-hasbg .s-ctner .s-ctner-menus .menus-ctner{height:286px;height:285px\0;*height:285px;_height:286px;*z-index:10}
.s-skin-hasbg .s-ctner-menus .s-menu{color:#fff;text-shadow:0 0 5px #000;background:0 0;background-image:-webkit-gradient(linear,left top,left bottom,from(rgba(0,0,0,.4)),to(rgba(0,0,0,.4)));background-image:-moz-linear-gradient(rgba(0,0,0,.4) 0,rgba(0,0,0,.4) 100%);background-image:-ms-linear-gradient(rgba(0,0,0,.4) 0,rgba(0,0,0,.4) 100%);background-image:-o-linear-gradient(rgba(0,0,0,.4) 0,rgba(0,0,0,.4) 100%);background-image:linear-gradient(rgba(0,0,0,.4) 0,rgba(0,0,0,.4) 100%);filter:progid:DXImageTransform.Microsoft.gradient(startColorstr='#66000000', endColorstr='#66000000')}
.s-skin-hasbg .s-ctner-menus .s-menu:hover{background:0 0!important;background-image:-webkit-gradient(linear,left top,left bottom,from(rgba(138,138,138,.6)),to(rgba(138,138,138,.6)))!important;background-image:-moz-linear-gradient(rgba(138,138,138,.6) 0,rgba(138,138,138,.6) 100%)!important;background-image:-ms-linear-gradient(rgba(138,138,138,.6) 0,rgba(138,138,138,.6) 100%)!important;background-image:-o-linear-gradient(rgba(138,138,138,.6) 0,rgba(138,138,138,.6) 100%)!important;background-image:linear-gradient(rgba(138,138,138,.6) 0,rgba(138,138,138,.6) 100%)!important;filter:progid:DXImageTransform.Microsoft.gradient(startColorstr='#998a8a8a', endColorstr='#998a8a8a')!important;text-shadow:none}
.s-skin-hasbg .s-ctner-menus .s-menu.add span{background:url(../img/skin_dark-48fa9dbf5c.png) no-repeat -688px -406px;_background:url(../img/skin_dark_ie-8647a11021.png) no-repeat -688px -406px}
.s-skin-hasbg .s-ctner-menus .s-menu.add.current span,.s-skin-hasbg .s-ctner-menus .s-menu.add:hover span{background-position:-543px -406px}
.s-skin-hasbg .s-ctner .s-ctner-menus .s-menu-fill{background:0 0;background-image:-webkit-gradient(linear,left top,left bottom,from(rgba(0,0,0,.4)),to(rgba(0,0,0,.4)));background-image:-moz-linear-gradient(rgba(0,0,0,.4) 0,rgba(0,0,0,.4) 100%);background-image:-ms-linear-gradient(rgba(0,0,0,.4) 0,rgba(0,0,0,.4) 100%);background-image:-o-linear-gradient(rgba(0,0,0,.4) 0,rgba(0,0,0,.4) 100%);background-image:linear-gradient(rgba(0,0,0,.4) 0,rgba(0,0,0,.4) 100%);filter:progid:DXImageTransform.Microsoft.gradient(startColorstr='#66000000', endColorstr='#66000000')}
.s-skin-hasbg .s-ctner-menus .top-border{background:0 0;background-image:-webkit-gradient(linear,left top,left bottom,from(rgba(0,0,0,.45)),to(rgba(0,0,0,0)));background-image:-moz-linear-gradient(rgba(0,0,0,.45) 0,rgba(0,0,0,0) 100%);background-image:-ms-linear-gradient(rgba(0,0,0,.45) 0,rgba(0,0,0,0) 100%);background-image:-o-linear-gradient(rgba(0,0,0,.45) 0,rgba(0,0,0,0) 100%);background-image:linear-gradient(rgba(0,0,0,.45) 0,rgba(0,0,0,0) 100%);filter:progid:DXImageTransform.Microsoft.gradient(startColorstr='#72000000', endColorstr='#00000000')}
.s-skin-hasbg .s-ctner-menus .bottom-border{background:0 0;background-image:-webkit-gradient(linear,left top,left bottom,from(rgba(0,0,0,0)),to(rgba(0,0,0,.45)));background-image:-moz-linear-gradient(rgba(0,0,0,0) 0,rgba(0,0,0,.45) 100%);background-image:-ms-linear-gradient(rgba(0,0,0,0) 0,rgba(0,0,0,.45) 100%);background-image:-o-linear-gradient(rgba(0,0,0,0) 0,rgba(0,0,0,.45) 100%);background-image:linear-gradient(rgba(0,0,0,0) 0,rgba(0,0,0,.45) 100%);filter:progid:DXImageTransform.Microsoft.gradient(startColorstr='#00000000', endColorstr='#72000000')}
.s-skin-hasbg .s-ctner .s-ctner-contents .s-content{border:none}
.s-skin-hasbg .s-ctner .s-ctner-contents .s-content-skin-bg{border:none;width:811px;height:318px;top:0;left:0;position:absolute;box-shadow:2px 2px 2px rgba(0,0,0,.07);-moz-box-shadow:2px 2px 2px rgba(0,0,0,.07);-webkit-box-shadow:2px 2px 2px rgba(0,0,0,.07);background:#fff;opacity:.1;filter:progid:DXImageTransform.Microsoft.Shadow(Strength=2, Direction=135, Color='#000000') alpha(opacity=10);-ms-filter:progid:DXImageTransform.Microsoft.Shadow(Strength=2, Direction=135, Color='#000000') alpha(opacity=10)}
.s-skin-hasbg .s-add-bar .s-add-bar-bg{background-color:#000;opacity:.5;filter:alpha(opacity=50)}
.s-skin-hasbg .s-add-bar a{background:url(../img/s_add_bar_skin-de741ff621.png) no-repeat 5px 114px;_background:url(../img/s_add_bar_skin_ie-0c2ae0a4ab.png) no-repeat 5px 114px}
.s-select{width:129px;height:27px;position:relative;display:inline-block}
.s-select-input,.s-select-input-mask{width:114px;color:#999;border:1px solid #ccc;border-color:#d3d3d3 #ebebeb #ebebeb #d3d3d3;height:25px;line-height:0;padding:0 7px 0 4px;padding:0 7px 0 4px\9;font:12px arial;outline:0;background:url(../img/input_bg-9af3291dff.png) no-repeat 0 0 #fff;overflow:hidden;cursor:default}
.s-select-input{position:absolute;left:-100000px}
.s-select-input-mask{position:absolute;padding-right:0;line-height:25px;text-align:left;top:0;*top:1px;_top:0;left:0;padding-right:19px;width:102px;z-index:1;white-space:nowrap;text-overflow:ellipsis;-o-text-overflow:ellipsis}
.s-select-arrow{border-left:1px solid #eee;outline:0;width:14px;height:25px;position:absolute;right:2px;_right:-5px;top:0;display:inline-block;background:url(../img/menu_bg-454e0b81f1.png) no-repeat -179px -264px;z-index:2}
.s-select-layer{position:absolute;width:125px;display:none;border:1px solid #ebebeb;border-top:none;top:27px;left:0;background:#fff;overflow-y:auto;overflow-x:hidden;z-index:1000}
.s-select-layer.show{display:block}
.s-select-layer .s-select-layer-option{width:102px;height:20px;line-height:20px;padding:0 19px 0 4px;text-align:left;overflow:hidden;white-space:nowrap;text-overflow:ellipsis;-o-text-overflow:ellipsis;font:12px;color:#666;cursor:default}
.s-select-layer .s-select-layer-option.current{background-color:#e2eaff;color:#000}
#s_wrap .s-superman-guide-mask{position:absolute;width:100%;height:100%;left:0;top:0;z-index:498;opacity:0;filter:alpha(opacity=0);background:#fff}
#s_wrap .s-index-guide-content{display:none;position:absolute;width:286px;height:87px;background:url(../img/superman_index_guide-be7fc8550c.png) no-repeat 0 0;_background:url(../img/superman_index_guide_ie6-d8e491b075.png) no-repeat 0 0;z-index:499;box-shadow:5px 5px 5px rgba(0,0,0,.25)}
#s_wrap .s-index-guide-content .arrow{position:absolute;top:-28px;left:89px;width:32px;height:28px;background:url(../img/superman_index_guide-be7fc8550c.png) no-repeat;_background:url(../img/superman_index_guide_ie6-d8e491b075.png) no-repeat 0 0}
#s_wrap .guide-card-step-1,#s_wrap .guide-water-step-1{top:65px;left:456px}
#s_wrap .guide-card-step-2,#s_wrap .guide-water-step-2{top:65px;left:1247px}
#s_wrap .guide-card-step-3,#s_wrap .guide-water-step-3{bottom:76px;left:542px}
#s_wrap .guide-card-step-1 .arrow,#s_wrap .guide-water-step-1 .arrow{background-position:-6px -94px}
#s_wrap .guide-card-step-2 .arrow,#s_wrap .guide-water-step-2 .arrow{background-position:-6px -94px}
#s_wrap .guide-card-step-3 .arrow,#s_wrap .guide-water-step-3 .arrow{top:26px;left:-36px;background-position:-43px -92px}
#s_wrap .guide-card-step-4 .arrow{top:60px;left:-36px;background-position:-87px -92px}
#s_wrap .s-index-guide-content .text{position:absolute;top:16px;left:16px;width:245px;height:20px;background:url(../img/superman_index_guide-be7fc8550c.png) no-repeat 0 0;_background:url(../img/superman_index_guide_ie6-d8e491b075.png) no-repeat 0 0}
#s_wrap .guide-card-step-1 .text,#s_wrap .guide-water-step-1 .text{background-position:-1px -173px}
#s_wrap .guide-card-step-2 .text,#s_wrap .guide-water-step-2 .text{background-position:-1px -194px}
#s_wrap .guide-card-step-3 .text,#s_wrap .guide-water-step-3 .text{background-position:-1px -217px}
#s_wrap .guide-card-step-4 .text{background-position:-1px -241px}
#s_wrap .s-index-guide-content .step{position:absolute;top:55px;left:77px;width:51px;height:6px;font-size:0;background:url(../img/superman_index_guide-be7fc8550c.png) no-repeat 0 0;_background:url(../img/superman_index_guide_ie6-d8e491b075.png) no-repeat 0 0}
#s_wrap .guide-water-step-1 .step{background-position:-147px -92px}
#s_wrap .guide-water-step-2 .step{background-position:-147px -104px}
#s_wrap .guide-water-step-3 .step{background-position:-147px -117px}
#s_wrap .guide-card-step-1 .step{background-position:-199px -92px}
#s_wrap .guide-card-step-2 .step{background-position:-199px -104px}
#s_wrap .guide-card-step-3 .step{background-position:-199px -117px}
#s_wrap .guide-card-step-4 .step{background-position:-3px -128px}
#s_wrap .s-index-guide-content .button{position:absolute;top:46px;left:138px;width:59px;height:23px;background:url(../img/superman_index_guide-be7fc8550c.png) no-repeat -194px -143px;_background:url(../img/superman_index_guide_ie6-d8e491b075.png) no-repeat -194px -143px}
#s_wrap .guide-card-step-2 .button,#s_wrap .guide-water-step-1 .button{background-position:-194px -143px}
#s_wrap .guide-card-step-4 .button,#s_wrap .guide-water-step-3 .button{background-position:-66px -143px}
#s_wrap .s-index-guide-content .close{position:absolute;top:10px;right:10px;width:16px;height:16px;cursor:pointer;background:url(../img/superman_index_guide-be7fc8550c.png) no-repeat -124px -94px;_background:url(../img/superman_index_guide_ie6-d8e491b075.png) no-repeat -124px -94px}
#s_wrap .guide-card-step-1 .button:hover,#s_wrap .guide-card-step-2 .button:hover,#s_wrap .guide-card-step-3 .button:hover,#s_wrap .guide-water-step-1 .button:hover,#s_wrap .guide-water-step-2 .button:hover{background-position:-130px -143px}
#s_wrap .guide-card-step-4 .button:hover,#s_wrap .guide-water-step-3 .button:hover{background-position:-3px -143px}
#s_wrap .s-index-guide-content .close:hover{background-position:-124px -112px}</style><style type="text/css" index="user_quit_dialog">.sui-dialog-mask{opacity:.2;filter:alpha(opacity=20)}
.s-quit-dialog{background:#fff;box-shadow:0 2px 10px 0 rgba(0,0,0,.1);border-radius:16px}
.s-quit-dialog .sui-dialog-body{padding:0}
.s-quit-dialog .dialog-con-wrap{padding:19px 24px 24px;position:relative}
.s-quit-dialog .dialog-close{position:absolute;cursor:pointer;top:8px;right:8px;height:20px;width:20px;line-height:20px;color:#d7d9e0;text-align:center;font-size:16px}
.s-quit-dialog .dialog-close:hover{color:#315efb}
.s-quit-dialog .dialog-title{margin-bottom:6px}
.s-quit-dialog .dialog-feedback-checkbox{margin-top:6px;vertical-align:top}
.s-quit-dialog .dialog-foot{height:30px}
.s-quit-dialog .dialog-foot-right{float:right}
.s-quit-dialog .dialog-feedback-btn{height:30px;line-height:30px;float:left;cursor:pointer}
.s-quit-dialog .dialog-textarea{height:84px}
.s-quit-dialog .dialog-textarea,.s-quit-dialog .dialog-user-contact{width:346px}
.s-quit-dialog .dialog-detailcon{display:none}
.s-quit-dialog .dialog-con-title1{margin-top:14px}
.s-quit-dialog .dialog-con-title2{margin-top:9px}
.s-quit-dialog .dialog-con-title{margin-top:19px}
.s-quit-dialog .dialog-input-wrap{position:relative;height:30px}
.s-quit-dialog .dialog-text-wrap{position:relative;height:96px}
.s-quit-dialog .dialog-input-tip,.s-quit-dialog .dialog-textarea-tip{display:none}
.s-quit-dialog .dialog-check-item{display:inline-block;width:183px;height:30px;line-height:30px;cursor:pointer;zoom:1}
.s-quit-dialog .dialog-check-item .text{vertical-align:middle}
.sui-wraper .sui-dialog-quite{position:fixed;_position:absolute}
#passport-login-pop{text-align:left}</style><style type="text/css">#s_main{display:block;position:relative;margin:0 auto;border-radius:16px;box-sizing:content-box;background:#fff;padding:14px 32px 0}
.new_input_superman #s_main.new-input2-main{margin-top:42px}
.s-skin-hasbg #s_main{background:rgba(255,255,255,.85)}
.main a,.main span{outline:0}
.s-mancacrd-main{position:relative;z-index:2}
.s-content{padding-bottom:24px}
#s_content_1{padding-bottom:0}
.s-menu-container,.s-menu-gurd{width:100%}
.s-menu-container{position:relative;min-height:32px}
.s-ctner-menus{position:relative;z-index:0;width:100%;margin:0 auto;text-align:left;height:32px}
#s_menu_gurd.s-down .s-ctner-menus{width:1040px}
@media screen and (max-width:1340px){
#s_menu_gurd.s-down .s-ctner-menus{width:896px}
}
#s_menu_gurd.s-down{background-color:#f5f5f6;margin-top:-32px}
.s-ctner-menus .s-menus-outer{width:100%}
.desktop-opt{display:none;position:absolute;z-index:3;right:0;top:-42px;color:#333;cursor:pointer}
.desktop-opt:hover{color:#315efb}
.desktop-opt:hover i{color:#315efb}
.desktop-opt i{position:relative;top:1px;left:0;margin-right:5px;font-size:16px;color:#333}
#s_xmancard_desktop{position:relative;min-width:944px}
.c-wrapper-hot-news-all #s_xmancard_desktop{min-width:auto}
.desktop-vis{display:block}
.s-ctner-contents.desktop-ctner-contents{min-height:auto}
.s-ctner-contents.desktop-ctner-contents .s-ctner-contents{min-height:auto}
.s-ctner-contents.desktop-ctner-contents .s-ctner-contents .s-content{padding-bottom:0}
.c-wrapper-hot-news-all .s-ctner-contents{min-height:0}
.c-wrapper-hot-news-all .s-ctner-contents .s-content{padding-bottom:15px}
.s-ctner-menus .menus-wrapper{height:100%;color:#9195a3}
.s-ctner-menus .s-menu-item{position:relative;display:inline-block;cursor:pointer;margin-right:24px;margin-top:4px;line-height:26px}
.s-top-nav .s-menu-gurd{min-height:40px}
.s-top-nav .s-ctner-menus{height:40px}
.s-top-nav .s-ctner-menus .s-menu-item{margin-top:2px;line-height:36px}
.s-ctner-menus .s-menu-item:hover{color:#333}
.s-ctner-menus .current,.s-ctner-menus .current:hover{color:#333}
.s-ctner-menus .s-menu-item-underline{display:none;content:'';position:absolute;bottom:-2px;left:0;height:2px;width:100%;background:#4e71f2;border-radius:4px}
.s-ctner-menus-ai .s-menu-item{font-size:16px;margin-top:0}
.s-ctner-menus-ai .s-menu-item.current{font-weight:600}
.s-ctner-menus-ai .s-menu-item-underline{bottom:-6px}
.s-ctner-menus .current .s-menu-item-underline{display:inline-block}
.desktop_bar .s-ctner-menus .s-menu-item-underline{background:#4e71f2}
.s-ctner-contents{width:100%;min-height:363px;_height:363px;_overflow:visible}
.s-ctner-contents-additem{min-height:634px}
.s-skin-hasbg .s-ctner-contents{_background-color:#fff;_border:none;_padding:1px}
.s-ctner-contents .s-content-load{padding:150px 0}
.s-ctner-contents .s-content-load img{width:44px;height:16px}
.s-ctner-contents .s-loading{height:53px;width:95%;padding-top:11px;margin:0 auto;border-top:1px solid #f0f0f0;font-size:14px}
.s-ctner-contents .s-loading img{width:44px;height:16px}
.s-ctner-contents .s-loading .load-done,.s-ctner-contents .s-loading .load-text,.s-ctner-contents .s-loading img{display:inline-block;*display:inline;zoom:1;vertical-align:middle}
.s-ctner-contents .s-loading .no-more{color:#9195a3;margin-top:10px}
.s-ctner-contents .s-loading .load-done{height:22px;width:22px;margin-top:10px;background:url('https://pss.bdstatic.com/static/superman/img/card_setts-c75091e958.png') no-repeat 0 -72px;_background:url('https://pss.bdstatic.com/static/superman/img/card_setts_ie-50c159f293.png') no-repeat 0 -72px}
.s-ctner-contents .s-loading .load-text{margin-left:10px;color:#9195a3}
.s-more-bar{width:90px;height:45px;cursor:pointer;position:absolute;bottom:-48px;left:50%;z-index:5;margin-left:-45px;padding-top:0}
.s-skin-hasbg .s-more-bar{bottom:-50px}
.s-ctner-contents .rect .animate-arrow{display:none}
.s-more-bar .rect{height:24px;width:34px;margin:0 auto;font-size:19px}
.s-card-split{height:363px;overflow:hidden}
.auto-refresh-toast-wrapper{width:560px;z-index:10000}
.bd-pc-toast-wrapper{z-index:10000}
.bd-pc-toast-wrapper .bd-pc-toast{position:absolute;max-width:210px;left:50%;top:50%;transform:translate(-50%,-50%);color:#fff;box-sizing:border-box;padding:6px 12px;border-radius:8px;border:0;font-size:13px;min-height:34px;line-height:21px;background:#626675}
.bd-pc-toast-wrapper .bd-pc-toast-icon{display:inline-block;position:relative;top:9px;width:14px;height:14px;margin-right:4px;background-position:0 0;background-size:cover;background-repeat:no-repeat;transform:translateY(-50%)}
.bd-pc-toast-wrapper .bd-pc-toast-icon.hide-icon{display:none}
.auto-refresh-toast{color:#fff;padding:10px 17px;border-radius:17px;border:0;font-size:13px;height:34px;line-height:34px;background:rgba(98,102,117,.95)}
.auto-refresh-toast.loading .auto-refresh-icon{background-image:url(https://b.bdstatic.com/searchbox/icms/searchbox/img/auto-refresh-loading.gif)}
.auto-refresh-toast.suc .auto-refresh-icon{background-image:url(https://b.bdstatic.com/searchbox/icms/searchbox/img/auto-refresh-suc.png)}
.auto-refresh-toast.retry .auto-refresh-icon{background-image:url(https://b.bdstatic.com/searchbox/icms/searchbox/img/auto-refresh-retry.png)}
.auto-refresh-toast.retry:hover{background:rgba(75,78,88,.95);cursor:pointer}
.auto-refresh-icon{display:inline-block;position:relative;top:9px;width:14px;height:14px;margin-right:4px;background-position:0 0;background-size:cover;background-repeat:no-repeat;transform:translateY(-50%)}
.s-auto-refresh-loading{color:#9195a3;font-size:13px;line-height:1;margin-top:20px}
#s_side_wrapper{position:fixed;right:24px;bottom:64px;background-color:#fbfbfb;width:44px;border-radius:22px;z-index:10}
#s_side_wrapper .side-entry{width:44px;height:44px;border-radius:50%}
#s_side_wrapper .side-entry .toast{display:none;box-shadow:0 1px 10px 0 rgba(0,0,0,.1);background:#fff;border-radius:6px;position:absolute;right:52px;height:34px;line-height:34px;font-size:13px;top:9px}
#s_side_wrapper .aging-entry{line-height:44px;cursor:pointer;box-sizing:border-box;padding-top:12px;position:relative}
#s_side_wrapper .aging-entry .toast{width:84px}
#s_side_wrapper .aging-entry .aging-entry-inner{width:18px;height:18px;margin:0 auto;background:url(data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAMAAABEpIrGAAAA7VBMVEUAAAD////////s8v////+txP+qwv+4zf/w9f/2+P+hu//Q3f+yyP+4zf/Q3f////+kvv+90P+80f+2yv/S4P/T4P/M2//z9/+cuP/V4P9Whv////9Uhf9Sg/9Pgf9NgP/8/f9di/9Xh/9lkf5aif9qlP7z9//k7P/c5v+2y/94nv51nP6lv/+LrP6Ep/6BpP5gjf7v9P+wxv/U4f/M2/+sxP+vxv73+f/P3v/J2v+5zf+ivP+fuv9xmf+Ytv6Usv6Hqf58of5vl/7m7v/g6f+zyf6QsP75+//q8P/B0v/W4//C1P6+0P6qwv6ct/76fHZiAAAAGnRSTlMAGAaVR/Py45aC9Mfy2b8t9OPZ2ce/v4L0x/e74/EAAAIZSURBVDjLZVPXYuIwEDSmQ4BLv5O0ku3Yhwu2IZTQe0hy7f8/57QSoYR5sVea1c424wgzl324LRRuH7I507hEJluYucCFEOBGhWzmy7X5+N0WwIjTbrcdBsKulM0z96onGCGE2X6n+cTkj/CqJ480igzkNXp26E9JkABSbBz8i4Bn3EkH840mKHoxs49fZQzt2Kd03FQEzSB3WsejB9Jqf1CJQBM0wCurABWBoub0gkDENwyStTHA62pwSWDtklRQ4FLfjnaiPqVW60hAYeLKNHIREOZuKTL80H6XBFCwn4BAmDOyLiOQUIlOSEjaoS+Ju57NZuul73Fml4w6yAivSLBW3MGfcfBmIegmArg3alICdJHgy1jQt8Z/6CcC4DdGXhLIoiWRACpbLYbDYW80GnXp2GH8ShP+PUvEoHsAIFq9Xm8+kXlIwkkI9pm+05Tm3yWqu9EiB0pkwjWBx2i+tND1XqeZqpPU4VhUbq/ekR8CwTRVoRxf3ifTbeIwcONNsJZ2lxFVKDMv1KNvS2zXdrnD+COvR1PQpTZKNlKD3cLCOJNnivgVxkw169BunlKFaV9/B+LQbqOsByY4IVgDB59dl/cjR9TIJV1Lh7CGqUqH/DDPhhZYOPkdLz6m0X7GrzPHsSe6zJwzxvm+5NeNi8U5ABfn7mz7zHJFrZ6+BY6rd7m8kQtcAtwwXzq4n69/vZbP1+pn6/8fsrRmHUhmpYYAAAAASUVORK5CYII=) no-repeat;background-size:16px;background-position:center;border-radius:50%;border:1px solid #626675}
#s_side_wrapper .aging-entry:hover{box-shadow:0 3px 5px 0 rgba(0,0,0,.1)}
#s_side_wrapper .aging-entry:hover .toast{display:block}
#s_side_wrapper .aging-entry:hover .aging-entry-inner{border:1px solid #4e6ef2}
#s_side_wrapper .qrcode-feed:hover,#s_side_wrapper .top-feed:hover{box-shadow:0 3px 5px 0 rgba(0,0,0,.1)}
#s_side_wrapper .qrcode-feed:hover .tooltip,#s_side_wrapper .top-feed:hover .tooltip{display:block}
#s_side_wrapper .qrcode-feed:hover .icon-mask-wrapper .icon,#s_side_wrapper .top-feed:hover .icon-mask-wrapper .icon{display:none}
#s_side_wrapper .qrcode-feed:hover .icon-mask-wrapper .icon-hover,#s_side_wrapper .top-feed:hover .icon-mask-wrapper .icon-hover{display:block}
#s_side_wrapper .icon-mask-wrapper{width:100%;border-radius:50%;padding:10px 0;cursor:pointer}
#s_side_wrapper .icon-mask-wrapper .icon,#s_side_wrapper .icon-mask-wrapper .icon-hover{height:24px;width:24px;margin-right:auto;margin-left:auto;display:block}
#s_side_wrapper .icon-mask-wrapper .icon-hover{display:none}
#s_side_wrapper .qrcode-feed{position:relative}
#s_side_wrapper .qrcode-feed:hover .icon-mask-wrapper::before{position:absolute;top:0;left:-20px;content:'';width:44px;height:44px}
#s_side_wrapper .tooltip{display:none;position:absolute;right:56px;bottom:0;background-color:#fff;z-index:303;box-shadow:0 2px 10px 0 rgba(0,0,0,.1);border-radius:12px}
#s_side_wrapper .totop-tooltip{width:80px;height:36px;line-height:36px}
#s_side_wrapper .qrcode-tooltip{width:279px;height:102px}
#s_side_wrapper .qrcode-tooltip .text{text-align:left;margin-top:24px;margin-left:16px}
#s_side_wrapper .qrcode-tooltip .download-text{color:#000}
#s_side_wrapper .qrcode-tooltip .qrcode{position:absolute;width:70px;height:70px;top:16px;right:16px}
@media screen and (max-width:1158px){
#s_side_wrapper{display:none}
}</style><style type="text/css" index="hotsearch">.s-hotsearch-wrapper{width:654px;margin:45px auto 0}
.s-hotsearch-wrapper.s-hotsearch-wrapper.s-hotsearch-wrapper.s-hotsearch-wrapper-login{position:relative;width:100%;margin-top:0;margin-bottom:-10px;padding:18px 0 0}
.s-hotsearch-wrapper.s-hotsearch-wrapper.s-hotsearch-wrapper.s-hotsearch-wrapper-login:after{content:'';display:block;clear:both}
.s-hotsearch-wrapper.s-hotsearch-wrapper.s-hotsearch-wrapper.s-hotsearch-wrapper-login .s-hotsearch-title{position:absolute;width:90px;top:-25px;right:0}
.s-hotsearch-wrapper.s-hotsearch-wrapper.s-hotsearch-wrapper.s-hotsearch-wrapper-login .s-hotsearch-title .hot-title{position:absolute;left:-484px;top:-2px}
.s-hotsearch-wrapper.s-hotsearch-wrapper.s-hotsearch-wrapper.s-hotsearch-wrapper-login .s-hotsearch-title .hot-title .hot-title-icon{visibility:hidden}
.s-hotsearch-wrapper.s-hotsearch-wrapper.s-hotsearch-wrapper.s-hotsearch-wrapper-login .s-hotsearch-title .hot-refresh{top:-3px}
.s-hotsearch-wrapper.hide{display:none}
.s-hotsearch-wrapper .s-hotsearch-title{height:24px;width:100%}
.s-hotsearch-wrapper .s-hotsearch-title .title-text{float:left;user-select:none}
.s-hotsearch-wrapper .s-hotsearch-title .title-text .arrow{color:#9195a3;font-size:15px}
.s-hotsearch-wrapper .s-hotsearch-title .title-text-nohover{float:left;user-select:none}
.s-hotsearch-wrapper .s-hotsearch-title .title-text-nohover .arrow{color:#9195a3;font-size:15px}
.s-hotsearch-wrapper .s-hotsearch-title .title-text:hover{color:#315efb}
.s-hotsearch-wrapper .s-hotsearch-title .title-text:hover .arrow{color:#315efb}
.s-hotsearch-wrapper .s-hotsearch-title .hot-refresh{cursor:pointer;user-select:none;float:right;height:16px;position:relative;top:1px}
.s-hotsearch-wrapper .s-hotsearch-title .hot-refresh .c-icon{font-size:16px;display:inline-block;position:relative;top:1px;color:#626675}
.s-hotsearch-wrapper .s-hotsearch-title .hot-refresh .c-icon.refresh-icon{width:16px;height:16px;line-height:16px}
.s-hotsearch-wrapper .s-hotsearch-title .hot-refresh .c-icon.rotate{-webkit-animation:rotate .2s linear forwards;-moz-animation:rotate .2s linear forwards;-o-animation:rotate .2s linear forwards;animation:rotate .2s linear forwards}
@keyframes rotate{
0%{transform:rotateZ(0)}
100%{transform:rotateZ(180deg)}
}
@-webkit-keyframes rotate{
0%{transform:rotateZ(0)}
100%{transform:rotateZ(180deg)}
}
@-moz-keyframes rotate{
0%{transform:rotateZ(0)}
100%{transform:rotateZ(180deg)}
}
@-o-keyframes rotate{
0%{transform:rotateZ(0)}
100%{transform:rotateZ(180deg)}
}
.s-hotsearch-wrapper .s-hotsearch-title .hot-refresh .hot-refresh-text{font-size:14px;line-height:14px;color:#626675;margin-left:2px}
.s-hotsearch-wrapper .s-hotsearch-title .hot-refresh:hover{color:#315efb;text-decoration:none}
.s-hotsearch-wrapper .s-hotsearch-title .hot-refresh:hover .c-icon,.s-hotsearch-wrapper .s-hotsearch-title .hot-refresh:hover .hot-refresh-text{color:#315efb}
.s-hotsearch-wrapper .s-hotsearch-content{text-align:left}
.s-hotsearch-wrapper .s-hotsearch-content .hotsearch-item{width:306px;float:left;height:36px;line-height:36px;white-space:nowrap}
.s-hotsearch-wrapper .s-hotsearch-content .hotsearch-item.odd{margin-right:20px;clear:both}
.s-hotsearch-wrapper .s-hotsearch-content .hotsearch-item.even{margin-left:20px}
.s-hotsearch-wrapper .s-hotsearch-content .title-content{float:left;height:36px;line-height:36px;width:100%}
.s-hotsearch-wrapper .s-hotsearch-content .title-content .title-content-index{margin-right:4px;width:22px;font-family:Arial,sans-serif;font-size:18px;line-height:18px;position:relative;top:1px}
.s-hotsearch-wrapper .s-hotsearch-content .title-content .title-content-noindex{height:4px;width:4px;margin-right:19px;margin-left:3px;position:relative;top:-2px;border-radius:2px;background:#9195a3;display:inline-block;vertical-align:middle}
.s-hotsearch-wrapper .s-hotsearch-content .title-content .title-content-top-icon{transform:rotate(180deg);height:18px;width:18px;display:inline-block;vertical-align:middle;line-height:18px;position:relative;top:-2px;left:-3px;font-size:18px}
.s-hotsearch-wrapper .s-hotsearch-content .title-content .title-content-title{font-size:16px}
.s-hotsearch-wrapper .s-hotsearch-content .title-content .title-content-icon{display:inline-block;width:62px;height:16px;margin-top:-3px;margin-right:3px;vertical-align:middle}
@media \0screen{
.s-hotsearch-wrapper .s-hotsearch-content .ie-vertical{margin-top:3px}
}
@media screen and (min-width:0 \0){
.s-hotsearch-wrapper .s-hotsearch-content .ie-vertical{margin-top:3px}
}
@media screen and (-ms-high-contrast:active),(-ms-high-contrast:none){
.s-hotsearch-wrapper .s-hotsearch-content .ie-vertical{margin-top:3px}
}
.s-hotsearch-wrapper .s-hotsearch-content .tag-width{max-width:284px;width:auto}
.s-hotsearch-wrapper .s-hotsearch-content .title-content-mark{margin-left:6px;position:relative;top:-2px}
.s-hotsearch-wrapper .s-hotsearch-content .title-content:hover{color:#315efb;text-decoration:underline}
.s-hotsearch-wrapper .s-hotsearch-button{display:inline-block;position:relative;left:50%;transform:translateX(-50%);margin-top:9px;padding:var(--cos-space-md,7px);gap:var(--cos-space-2-xs,6px);text-decoration:none}
.s-hotsearch-wrapper .s-hotsearch-button .title-text .arrow{position:relative;top:-.5px;font-size:12px}
.s-hotsearch-wrapper .s-hotsearch-button .title-text:hover{color:#315efb}
.s-hotsearch-wrapper .s-hotsearch-button .title-text:hover .arrow{color:#315efb}
.new_input_superman .s-hotsearch-wrapper{margin-top:128px}
.new_input_superman .new-input2-main .s_form_nologin .s-hotsearch-wrapper{margin-top:131px}
.new_input_superman .new-input2-main .s_form_nologin .s-hotsearch-wrapper.have-search-guide{margin-top:207px}
.new_input_superman .s_form_nologin .s-hotsearch-wrapper{margin-top:90px}
.new_input_superman .s_form_nologin .s-hotsearch-wrapper.have-search-guide{margin-top:165px}
.s-hotboard-wrapper{margin-top:-58px}
.s-hotboard-wrapper.s-hot-board-wraper-config{margin-top:6px}
.s-hotboard-wrapper.hide{display:none}
.s-hotboard-wrapper.s-hot-board-recommend-style{margin-top:18px}
.s-ps-islite .s-hotsearch-wrapper{display:none}</style><script>
        (function(){
        var hashMatch=document.location.href.match(/#+(.*wd=(.+))/);
        if(hashMatch&&hashMatch[0]&&hashMatch[1]){
            // 隐藏页面内容
                var css='body {display: none}',
                    head=document.head||document.getElementsByTagName('head')[0],
                    style=document.createElement('style');

                if(style.styleSheet){
                    style.styleSheet.cssText = css;
                }else{
                    style.appendChild(document.createTextNode(css));
                }

                head.appendChild(style);
                if(hashMatch[2]){
                    var encodeQuery=hashMatch[2].replace(/&/g, '%26').replace(/\+/g, '%2B');
                    hashMatch[1]=hashMatch[1].split('wd=')[0]+'wd='+encodeQuery;
                }

            location.href="//"+location.host+"/s?"+hashMatch[1];
        }
        })();
        
    </script><script>
    
    

    
    

    window.sysTime=1743071185;
    _manCard = {
        asynJs : [],
        asynLoad : function(id){
            _manCard.asynJs.push(id);
        }
    };
    
    
    
        
    
    window.bds = {
        comm: {
            ishome: 1,
            qid: "",
            alwaysMonitor: "0"
        }
    };
</script><script type="application/json" id="topmenuloc-data" data-for="result-data">
    ""</script><script type="application/json" id="underbox-tips-data" data-for="result-data">
    {"isBdWordLinkRecallFromGuide":"","bdWordLinkGuideShowUrl":"","bdWordLinkGuideClickUrl":""}</script><script type="application/json" id="top-right-operate-data" data-for="result-data">
    ""</script><script type="application/json" id="placeholder-data" data-for="result-data">
    {"boxRecData":"","boxRecSwitch":"","hotData":[{"banner":[{"card_title":"高水平开放拓展国际合作新空间","ext":{"hotTags":0},"heat_score":7990475,"hotScore":"7990475","hotTag":"0","img":"https:\/\/fyb-2.cdn.bcebos.com\/hotboard_image\/9c7b6e1fe67a5a5af8be57918e6c076a","index":0,"linkurl":"https:\/\/www.baidu.com\/s?wd=%E9%AB%98%E6%B0%B4%E5%B9%B3%E5%BC%80%E6%94%BE%E6%8B%93%E5%B1%95%E5%9B%BD%E9%99%85%E5%90%88%E4%BD%9C%E6%96%B0%E7%A9%BA%E9%97%B4&sa=fyb_n_homepage&rsv_dl=fyb_n_homepage&from=super&cl=3&tn=baidutop10&fr=top1000&rsv_idx=2&hisfilter=1","pre_tag":"","query":"高水平开放拓展国际合作新空间","sa":"fyb_n_homepage","url":"https:\/\/www.baidu.com\/s?wd=%E9%AB%98%E6%B0%B4%E5%B9%B3%E5%BC%80%E6%94%BE%E6%8B%93%E5%B1%95%E5%9B%BD%E9%99%85%E5%90%88%E4%BD%9C%E6%96%B0%E7%A9%BA%E9%97%B4&sa=fyb_n_homepage&rsv_dl=fyb_n_homepage&from=super","word":"高水平开放拓展国际合作新空间"},{"card_title":"大桥垮塌致62人死亡失踪 43人被问责","ext":{"hotTags":4},"formatHotScore":"791万","heat_score":7913081,"hotScore":"7913081","hotTag":"4","img":"https:\/\/fyb-2.cdn.bcebos.com\/hotboard_image\/688b16bfec858fbafefea5f94af253bf","index":1,"linkurl":"https:\/\/www.baidu.com\/s?wd=%E5%A4%A7%E6%A1%A5%E5%9E%AE%E5%A1%8C%E8%87%B462%E4%BA%BA%E6%AD%BB%E4%BA%A1%E5%A4%B1%E8%B8%AA+43%E4%BA%BA%E8%A2%AB%E9%97%AE%E8%B4%A3&sa=fyb_n_homepage&rsv_dl=fyb_n_homepage&from=super&cl=3&tn=baidutop10&fr=top1000&rsv_idx=2&hisfilter=1","pre_tag":"","query":"大桥垮塌致62人死亡失踪 43人被问责","sa":"fyb_n_homepage","url":"https:\/\/www.baidu.com\/s?wd=%E5%A4%A7%E6%A1%A5%E5%9E%AE%E5%A1%8C%E8%87%B462%E4%BA%BA%E6%AD%BB%E4%BA%A1%E5%A4%B1%E8%B8%AA+43%E4%BA%BA%E8%A2%AB%E9%97%AE%E8%B4%A3&sa=fyb_n_homepage&rsv_dl=fyb_n_homepage&from=super","word":"大桥垮塌致62人死亡失踪 43人被问责"},{"card_title":"DeepSeek创始人首登全球富豪榜","ext":{"hotTags":3},"formatHotScore":"780万","heat_score":7806164,"hotScore":"7806164","hotTag":"3","img":"https:\/\/fyb-2.cdn.bcebos.com\/hotboard_image\/320a3eeb891c4ed08c322cb2f039ff5f","index":2,"linkurl":"https:\/\/www.baidu.com\/s?wd=DeepSeek%E5%88%9B%E5%A7%8B%E4%BA%BA%E9%A6%96%E7%99%BB%E5%85%A8%E7%90%83%E5%AF%8C%E8%B1%AA%E6%A6%9C&sa=fyb_n_homepage&rsv_dl=fyb_n_homepage&from=super&cl=3&tn=baidutop10&fr=top1000&rsv_idx=2&hisfilter=1","pre_tag":"","query":"DeepSeek创始人首登全球富豪榜","sa":"fyb_n_homepage","url":"https:\/\/www.baidu.com\/s?wd=DeepSeek%E5%88%9B%E5%A7%8B%E4%BA%BA%E9%A6%96%E7%99%BB%E5%85%A8%E7%90%83%E5%AF%8C%E8%B1%AA%E6%A6%9C&sa=fyb_n_homepage&rsv_dl=fyb_n_homepage&from=super","word":"DeepSeek创始人首登全球富豪榜"},{"card_title":"新一批食品安全国家标准来了","ext":{"hotTags":0},"formatHotScore":"776万","heat_score":7761903,"hotScore":"7761903","hotTag":"0","img":"https:\/\/fyb-2.cdn.bcebos.com\/hotboard_image\/39c9408a2389bb58238116c249469f9a","index":3,"linkurl":"https:\/\/www.baidu.com\/s?wd=%E6%96%B0%E4%B8%80%E6%89%B9%E9%A3%9F%E5%93%81%E5%AE%89%E5%85%A8%E5%9B%BD%E5%AE%B6%E6%A0%87%E5%87%86%E6%9D%A5%E4%BA%86&sa=fyb_n_homepage&rsv_dl=fyb_n_homepage&from=super&cl=3&tn=baidutop10&fr=top1000&rsv_idx=2&hisfilter=1","pre_tag":"","query":"新一批食品安全国家标准来了","sa":"fyb_n_homepage","url":"https:\/\/www.baidu.com\/s?wd=%E6%96%B0%E4%B8%80%E6%89%B9%E9%A3%9F%E5%93%81%E5%AE%89%E5%85%A8%E5%9B%BD%E5%AE%B6%E6%A0%87%E5%87%86%E6%9D%A5%E4%BA%86&sa=fyb_n_homepage&rsv_dl=fyb_n_homepage&from=super","word":"新一批食品安全国家标准来了"},{"card_title":"香飘飘奶茶涨价至16元一杯","ext":{"hotTags":3},"formatHotScore":"763万","heat_score":7633546,"hotScore":"7633546","hotTag":"3","img":"https:\/\/fyb-2.cdn.bcebos.com\/hotboard_image\/dfc3c82dc012d2ffa9fb8a5cdc6eed09","index":4,"linkurl":"https:\/\/www.baidu.com\/s?wd=%E9%A6%99%E9%A3%98%E9%A3%98%E5%A5%B6%E8%8C%B6%E6%B6%A8%E4%BB%B7%E8%87%B316%E5%85%83%E4%B8%80%E6%9D%AF&sa=fyb_n_homepage&rsv_dl=fyb_n_homepage&from=super&cl=3&tn=baidutop10&fr=top1000&rsv_idx=2&hisfilter=1","pre_tag":"","query":"香飘飘奶茶涨价至16元一杯","sa":"fyb_n_homepage","url":"https:\/\/www.baidu.com\/s?wd=%E9%A6%99%E9%A3%98%E9%A3%98%E5%A5%B6%E8%8C%B6%E6%B6%A8%E4%BB%B7%E8%87%B316%E5%85%83%E4%B8%80%E6%9D%AF&sa=fyb_n_homepage&rsv_dl=fyb_n_homepage&from=super","word":"香飘飘奶茶涨价至16元一杯"},{"card_title":"于正曝某男演员将女主戏几乎删没了","ext":{"hotTags":0},"extraInfo":"{\"extraParams\":\"AFD851bSI6MSwicHJvZHVjdElkIjoxOCwiX2NsaWVudF90eXBlIjowLCJuZXRfdHlwZSI6MCwiY3VpZCI6IiIsImVtcHR5X2FkIjoxLCJwcm9kdWN0X2lkIjoxOCwicGFzc3BvcnRJZCI6IkFGRDk9QT1NIiwidWlfZnJvbSI6NH0=c2MDkyMjg5MjY5OTg4OTc1NjAsImVpZF9saXN0IjoiIiwiZmVlZF9jb3VudCI6MCwiZnJlc2hUeXBlIjowLCJzZWFyY2hJZDIiOjc2MDkyMjg5MjY5OTg4OTc1NjAsImJhaWR1SWQiOiIiLCJzaXAiOiIwLjAuMC4wIiwicmVxX2eyJwcmljZXNvcnRfcSI6MCwibG9naWQiOiI0NDkzMjE2OTEiLCJwbGFjZV9pZCI6IjE2NDE1MzAxMDUzMDkiLCJpZGMiOiJnemh4eSIsIm5ldFR5cGUiOjAsInNlYXJjaF90aW1lIjoxNzQzMDcxMTM0LCJzZWFyY2hfaWQiOj\"}","feedFloorType":"2","formatHotScore":"751万","heat_score":7514493,"hotScore":"7514493","hotTag":"0","img":"https:\/\/fyb-2.cdn.bcebos.com\/hotboard_image\/4298a666ea81fe78909eb47670978170","index":5,"linkurl":"https:\/\/www.baidu.com\/s?wd=%E4%BA%8E%E6%AD%A3%E6%9B%9D%E6%9F%90%E7%94%B7%E6%BC%94%E5%91%98%E5%B0%86%E5%A5%B3%E4%B8%BB%E6%88%8F%E5%87%A0%E4%B9%8E%E5%88%A0%E6%B2%A1%E4%BA%86&sa=fyb_n_homepage&rsv_dl=fyb_n_homepage&from=super&cl=3&tn=baidutop10&fr=top1000&rsv_idx=2&hisfilter=1","pre_tag":"","query":"于正曝某男演员将女主戏几乎删没了","sa":"fyb_n_homepage","url":"https:\/\/www.baidu.com\/s?wd=%E4%BA%8E%E6%AD%A3%E6%9B%9D%E6%9F%90%E7%94%B7%E6%BC%94%E5%91%98%E5%B0%86%E5%A5%B3%E4%B8%BB%E6%88%8F%E5%87%A0%E4%B9%8E%E5%88%A0%E6%B2%A1%E4%BA%86&sa=fyb_n_homepage&rsv_dl=fyb_n_homepage&from=super","word":"于正曝某男演员将女主戏几乎删没了"},{"card_title":"“胖都来”成立 胖东来回应","ext":{"hotTags":0},"formatHotScore":"740万","heat_score":7407188,"hotScore":"7407188","hotTag":"0","img":"https:\/\/fyb-2.cdn.bcebos.com\/hotboard_image\/ec4e64cdcd1b0536b4cbb912c281fbcf","index":6,"linkurl":"https:\/\/www.baidu.com\/s?wd=%E2%80%9C%E8%83%96%E9%83%BD%E6%9D%A5%E2%80%9D%E6%88%90%E7%AB%8B+%E8%83%96%E4%B8%9C%E6%9D%A5%E5%9B%9E%E5%BA%94&sa=fyb_n_homepage&rsv_dl=fyb_n_homepage&from=super&cl=3&tn=baidutop10&fr=top1000&rsv_idx=2&hisfilter=1","pre_tag":"","query":"“胖都来”成立 胖东来回应","sa":"fyb_n_homepage","url":"https:\/\/www.baidu.com\/s?wd=%E2%80%9C%E8%83%96%E9%83%BD%E6%9D%A5%E2%80%9D%E6%88%90%E7%AB%8B+%E8%83%96%E4%B8%9C%E6%9D%A5%E5%9B%9E%E5%BA%94&sa=fyb_n_homepage&rsv_dl=fyb_n_homepage&from=super","word":"“胖都来”成立 胖东来回应"},{"card_title":"一汽奔腾悦意来了","ext":{"hotTags":0},"formatHotScore":"735万","heat_score":7358287,"hotScore":"7358287","hotTag":"0","img":"https:\/\/gips3.baidu.com\/it\/u=3805236591,2431602968&fm=3028&app=3028&f=PNG&fmt=auto&q=75&size=f352_234","index":7,"linkurl":"https:\/\/www.baidu.com\/s?wd=%E4%B8%80%E6%B1%BD%E5%A5%94%E8%85%BE%E6%82%A6%E6%84%8F%E6%9D%A5%E4%BA%86&sa=fyb_n_homepage&rsv_dl=fyb_n_homepage&from=super&cl=3&tn=baidutop10&fr=top1000&rsv_idx=2&hisfilter=1","pre_tag":"","query":"一汽奔腾悦意来了","sa":"fyb_n_homepage","url":"https:\/\/www.baidu.com\/s?wd=%E4%B8%80%E6%B1%BD%E5%A5%94%E8%85%BE%E6%82%A6%E6%84%8F%E6%9D%A5%E4%BA%86&sa=fyb_n_homepage&rsv_dl=fyb_n_homepage&from=super","word":"一汽奔腾悦意来了"},{"card_title":"外交部回应李嘉诚集团出售海外港口","ext":{"hotTags":0},"formatHotScore":"723万","heat_score":7232578,"hotScore":"7232578","hotTag":"0","img":"https:\/\/fyb-2.cdn.bcebos.com\/hotboard_image\/d290aab9e3ee7184f903d8fbc57efcb8","index":8,"linkurl":"https:\/\/www.baidu.com\/s?wd=%E5%A4%96%E4%BA%A4%E9%83%A8%E5%9B%9E%E5%BA%94%E6%9D%8E%E5%98%89%E8%AF%9A%E9%9B%86%E5%9B%A2%E5%87%BA%E5%94%AE%E6%B5%B7%E5%A4%96%E6%B8%AF%E5%8F%A3&sa=fyb_n_homepage&rsv_dl=fyb_n_homepage&from=super&cl=3&tn=baidutop10&fr=top1000&rsv_idx=2&hisfilter=1","pre_tag":"","query":"外交部回应李嘉诚集团出售海外港口","sa":"fyb_n_homepage","url":"https:\/\/www.baidu.com\/s?wd=%E5%A4%96%E4%BA%A4%E9%83%A8%E5%9B%9E%E5%BA%94%E6%9D%8E%E5%98%89%E8%AF%9A%E9%9B%86%E5%9B%A2%E5%87%BA%E5%94%AE%E6%B5%B7%E5%A4%96%E6%B8%AF%E5%8F%A3&sa=fyb_n_homepage&rsv_dl=fyb_n_homepage&from=super","word":"外交部回应李嘉诚集团出售海外港口"},{"card_title":"赵露思承认自己对生命没敬畏","ext":{"hotTags":0},"formatHotScore":"716万","heat_score":7164090,"hotScore":"7164090","hotTag":"0","img":"https:\/\/gips3.baidu.com\/it\/u=3805236591,2431602968&fm=3028&app=3028&f=PNG&fmt=auto&q=75&size=f352_234","index":9,"linkurl":"https:\/\/www.baidu.com\/s?wd=%E8%B5%B5%E9%9C%B2%E6%80%9D%E6%89%BF%E8%AE%A4%E8%87%AA%E5%B7%B1%E5%AF%B9%E7%94%9F%E5%91%BD%E6%B2%A1%E6%95%AC%E7%95%8F&sa=fyb_n_homepage&rsv_dl=fyb_n_homepage&from=super&cl=3&tn=baidutop10&fr=top1000&rsv_idx=2&hisfilter=1","pre_tag":"","query":"赵露思承认自己对生命没敬畏","sa":"fyb_n_homepage","url":"https:\/\/www.baidu.com\/s?wd=%E8%B5%B5%E9%9C%B2%E6%80%9D%E6%89%BF%E8%AE%A4%E8%87%AA%E5%B7%B1%E5%AF%B9%E7%94%9F%E5%91%BD%E6%B2%A1%E6%95%AC%E7%95%8F&sa=fyb_n_homepage&rsv_dl=fyb_n_homepage&from=super","word":"赵露思承认自己对生命没敬畏"},{"card_title":"爸爸去世女儿被判和4个姑姑平分遗产","ext":{"hotTags":3},"formatHotScore":"705万","heat_score":7057881,"hotScore":"7057881","hotTag":"3","img":"https:\/\/fyb-2.cdn.bcebos.com\/hotboard_image\/05368d9b9caaedab1e2b363466b8a57a","index":10,"linkurl":"https:\/\/www.baidu.com\/s?wd=%E7%88%B8%E7%88%B8%E5%8E%BB%E4%B8%96%E5%A5%B3%E5%84%BF%E8%A2%AB%E5%88%A4%E5%92%8C4%E4%B8%AA%E5%A7%91%E5%A7%91%E5%B9%B3%E5%88%86%E9%81%97%E4%BA%A7&sa=fyb_n_homepage&rsv_dl=fyb_n_homepage&from=super&cl=3&tn=baidutop10&fr=top1000&rsv_idx=2&hisfilter=1","pre_tag":"","query":"爸爸去世女儿被判和4个姑姑平分遗产","sa":"fyb_n_homepage","url":"https:\/\/www.baidu.com\/s?wd=%E7%88%B8%E7%88%B8%E5%8E%BB%E4%B8%96%E5%A5%B3%E5%84%BF%E8%A2%AB%E5%88%A4%E5%92%8C4%E4%B8%AA%E5%A7%91%E5%A7%91%E5%B9%B3%E5%88%86%E9%81%97%E4%BA%A7&sa=fyb_n_homepage&rsv_dl=fyb_n_homepage&from=super","word":"爸爸去世女儿被判和4个姑姑平分遗产"},{"card_title":"1人造谣“三亚变空城”被处罚","ext":{"hotTags":0},"formatHotScore":"698万","heat_score":6981671,"hotScore":"6981671","hotTag":"0","img":"https:\/\/fyb-2.cdn.bcebos.com\/hotboard_image\/1c88d02f256b4b0fcb404c7644db1168","index":11,"linkurl":"https:\/\/www.baidu.com\/s?wd=1%E4%BA%BA%E9%80%A0%E8%B0%A3%E2%80%9C%E4%B8%89%E4%BA%9A%E5%8F%98%E7%A9%BA%E5%9F%8E%E2%80%9D%E8%A2%AB%E5%A4%84%E7%BD%9A&sa=fyb_n_homepage&rsv_dl=fyb_n_homepage&from=super&cl=3&tn=baidutop10&fr=top1000&rsv_idx=2&hisfilter=1","pre_tag":"","query":"1人造谣“三亚变空城”被处罚","sa":"fyb_n_homepage","url":"https:\/\/www.baidu.com\/s?wd=1%E4%BA%BA%E9%80%A0%E8%B0%A3%E2%80%9C%E4%B8%89%E4%BA%9A%E5%8F%98%E7%A9%BA%E5%9F%8E%E2%80%9D%E8%A2%AB%E5%A4%84%E7%BD%9A&sa=fyb_n_homepage&rsv_dl=fyb_n_homepage&from=super","word":"1人造谣“三亚变空城”被处罚"},{"card_title":"儿子给腰突父亲买40小时硬座票","ext":{"hotTags":0},"formatHotScore":"680万","heat_score":6806746,"hotScore":"6806746","hotTag":"0","img":"https:\/\/gips3.baidu.com\/it\/u=3805236591,2431602968&fm=3028&app=3028&f=PNG&fmt=auto&q=75&size=f352_234","index":12,"linkurl":"https:\/\/www.baidu.com\/s?wd=%E5%84%BF%E5%AD%90%E7%BB%99%E8%85%B0%E7%AA%81%E7%88%B6%E4%BA%B2%E4%B9%B040%E5%B0%8F%E6%97%B6%E7%A1%AC%E5%BA%A7%E7%A5%A8&sa=fyb_n_homepage&rsv_dl=fyb_n_homepage&from=super&cl=3&tn=baidutop10&fr=top1000&rsv_idx=2&hisfilter=1","pre_tag":"","query":"儿子给腰突父亲买40小时硬座票","sa":"fyb_n_homepage","url":"https:\/\/www.baidu.com\/s?wd=%E5%84%BF%E5%AD%90%E7%BB%99%E8%85%B0%E7%AA%81%E7%88%B6%E4%BA%B2%E4%B9%B040%E5%B0%8F%E6%97%B6%E7%A1%AC%E5%BA%A7%E7%A5%A8&sa=fyb_n_homepage&rsv_dl=fyb_n_homepage&from=super","word":"儿子给腰突父亲买40小时硬座票"},{"card_title":"李在明被判无罪后跪地慰问山火灾民","ext":{"hotTags":0},"formatHotScore":"675万","heat_score":6750093,"hotScore":"6750093","hotTag":"0","img":"https:\/\/gips3.baidu.com\/it\/u=3805236591,2431602968&fm=3028&app=3028&f=PNG&fmt=auto&q=75&size=f352_234","index":13,"linkurl":"https:\/\/www.baidu.com\/s?wd=%E6%9D%8E%E5%9C%A8%E6%98%8E%E8%A2%AB%E5%88%A4%E6%97%A0%E7%BD%AA%E5%90%8E%E8%B7%AA%E5%9C%B0%E6%85%B0%E9%97%AE%E5%B1%B1%E7%81%AB%E7%81%BE%E6%B0%91&sa=fyb_n_homepage&rsv_dl=fyb_n_homepage&from=super&cl=3&tn=baidutop10&fr=top1000&rsv_idx=2&hisfilter=1","pre_tag":"","query":"李在明被判无罪后跪地慰问山火灾民","sa":"fyb_n_homepage","url":"https:\/\/www.baidu.com\/s?wd=%E6%9D%8E%E5%9C%A8%E6%98%8E%E8%A2%AB%E5%88%A4%E6%97%A0%E7%BD%AA%E5%90%8E%E8%B7%AA%E5%9C%B0%E6%85%B0%E9%97%AE%E5%B1%B1%E7%81%AB%E7%81%BE%E6%B0%91&sa=fyb_n_homepage&rsv_dl=fyb_n_homepage&from=super","word":"李在明被判无罪后跪地慰问山火灾民"},{"card_title":"凤凰传奇曾毅勇闯电影圈","ext":{"hotTags":0},"formatHotScore":"668万","heat_score":6685566,"hotScore":"6685566","hotTag":"0","img":"https:\/\/gips3.baidu.com\/it\/u=3805236591,2431602968&fm=3028&app=3028&f=PNG&fmt=auto&q=75&size=f352_234","index":14,"linkurl":"https:\/\/www.baidu.com\/s?wd=%E5%87%A4%E5%87%B0%E4%BC%A0%E5%A5%87%E6%9B%BE%E6%AF%85%E5%8B%87%E9%97%AF%E7%94%B5%E5%BD%B1%E5%9C%88&sa=fyb_n_homepage&rsv_dl=fyb_n_homepage&from=super&cl=3&tn=baidutop10&fr=top1000&rsv_idx=2&hisfilter=1","pre_tag":"","query":"凤凰传奇曾毅勇闯电影圈","sa":"fyb_n_homepage","url":"https:\/\/www.baidu.com\/s?wd=%E5%87%A4%E5%87%B0%E4%BC%A0%E5%A5%87%E6%9B%BE%E6%AF%85%E5%8B%87%E9%97%AF%E7%94%B5%E5%BD%B1%E5%9C%88&sa=fyb_n_homepage&rsv_dl=fyb_n_homepage&from=super","word":"凤凰传奇曾毅勇闯电影圈"},{"card_title":"杭州女童坠亡案雇主被保姆女儿起诉","ext":{"hotTags":0},"formatHotScore":"656万","heat_score":6562148,"hotScore":"6562148","hotTag":"0","img":"https:\/\/fyb-2.cdn.bcebos.com\/hotboard_image\/18237fe951eebd718eeb46362eafa860","index":15,"linkurl":"https:\/\/www.baidu.com\/s?wd=%E6%9D%AD%E5%B7%9E%E5%A5%B3%E7%AB%A5%E5%9D%A0%E4%BA%A1%E6%A1%88%E9%9B%87%E4%B8%BB%E8%A2%AB%E4%BF%9D%E5%A7%86%E5%A5%B3%E5%84%BF%E8%B5%B7%E8%AF%89&sa=fyb_n_homepage&rsv_dl=fyb_n_homepage&from=super&cl=3&tn=baidutop10&fr=top1000&rsv_idx=2&hisfilter=1","pre_tag":"","query":"杭州女童坠亡案雇主被保姆女儿起诉","sa":"fyb_n_homepage","url":"https:\/\/www.baidu.com\/s?wd=%E6%9D%AD%E5%B7%9E%E5%A5%B3%E7%AB%A5%E5%9D%A0%E4%BA%A1%E6%A1%88%E9%9B%87%E4%B8%BB%E8%A2%AB%E4%BF%9D%E5%A7%86%E5%A5%B3%E5%84%BF%E8%B5%B7%E8%AF%89&sa=fyb_n_homepage&rsv_dl=fyb_n_homepage&from=super","word":"杭州女童坠亡案雇主被保姆女儿起诉"},{"card_title":"你以为的毒贩vs你不知道的伪装","ext":{"hotTags":0},"formatHotScore":"649万","heat_score":6496673,"hotScore":"6496673","hotTag":"0","img":"https:\/\/gips3.baidu.com\/it\/u=3805236591,2431602968&fm=3028&app=3028&f=PNG&fmt=auto&q=75&size=f352_234","index":16,"linkurl":"https:\/\/www.baidu.com\/s?wd=%E4%BD%A0%E4%BB%A5%E4%B8%BA%E7%9A%84%E6%AF%92%E8%B4%A9vs%E4%BD%A0%E4%B8%8D%E7%9F%A5%E9%81%93%E7%9A%84%E4%BC%AA%E8%A3%85&sa=fyb_n_homepage&rsv_dl=fyb_n_homepage&from=super&cl=3&tn=baidutop10&fr=top1000&rsv_idx=2&hisfilter=1","pre_tag":"","query":"你以为的毒贩vs你不知道的伪装","sa":"fyb_n_homepage","url":"https:\/\/www.baidu.com\/s?wd=%E4%BD%A0%E4%BB%A5%E4%B8%BA%E7%9A%84%E6%AF%92%E8%B4%A9vs%E4%BD%A0%E4%B8%8D%E7%9F%A5%E9%81%93%E7%9A%84%E4%BC%AA%E8%A3%85&sa=fyb_n_homepage&rsv_dl=fyb_n_homepage&from=super","word":"你以为的毒贩vs你不知道的伪装"},{"card_title":"网红胡楚靓确诊多囊卵巢综合症","ext":{"hotTags":0},"formatHotScore":"639万","heat_score":6398232,"hotScore":"6398232","hotTag":"0","img":"https:\/\/fyb-2.cdn.bcebos.com\/hotboard_image\/ab15fd0187e123436483eae8349929a0","index":17,"linkurl":"https:\/\/www.baidu.com\/s?wd=%E7%BD%91%E7%BA%A2%E8%83%A1%E6%A5%9A%E9%9D%93%E7%A1%AE%E8%AF%8A%E5%A4%9A%E5%9B%8A%E5%8D%B5%E5%B7%A2%E7%BB%BC%E5%90%88%E7%97%87&sa=fyb_n_homepage&rsv_dl=fyb_n_homepage&from=super&cl=3&tn=baidutop10&fr=top1000&rsv_idx=2&hisfilter=1","pre_tag":"","query":"网红胡楚靓确诊多囊卵巢综合症","sa":"fyb_n_homepage","url":"https:\/\/www.baidu.com\/s?wd=%E7%BD%91%E7%BA%A2%E8%83%A1%E6%A5%9A%E9%9D%93%E7%A1%AE%E8%AF%8A%E5%A4%9A%E5%9B%8A%E5%8D%B5%E5%B7%A2%E7%BB%BC%E5%90%88%E7%97%87&sa=fyb_n_homepage&rsv_dl=fyb_n_homepage&from=super","word":"网红胡楚靓确诊多囊卵巢综合症"},{"card_title":"赚70万司机给妻子转200自责太少","ext":{"hotTags":0},"formatHotScore":"628万","heat_score":6288829,"hotScore":"6288829","hotTag":"0","img":"https:\/\/fyb-2.cdn.bcebos.com\/hotboard_image\/7694fb2c032443a3747a2f5d25e8d053","index":18,"linkurl":"https:\/\/www.baidu.com\/s?wd=%E8%B5%9A70%E4%B8%87%E5%8F%B8%E6%9C%BA%E7%BB%99%E5%A6%BB%E5%AD%90%E8%BD%AC200%E8%87%AA%E8%B4%A3%E5%A4%AA%E5%B0%91&sa=fyb_n_homepage&rsv_dl=fyb_n_homepage&from=super&cl=3&tn=baidutop10&fr=top1000&rsv_idx=2&hisfilter=1","pre_tag":"","query":"赚70万司机给妻子转200自责太少","sa":"fyb_n_homepage","url":"https:\/\/www.baidu.com\/s?wd=%E8%B5%9A70%E4%B8%87%E5%8F%B8%E6%9C%BA%E7%BB%99%E5%A6%BB%E5%AD%90%E8%BD%AC200%E8%87%AA%E8%B4%A3%E5%A4%AA%E5%B0%91&sa=fyb_n_homepage&rsv_dl=fyb_n_homepage&from=super","word":"赚70万司机给妻子转200自责太少"},{"card_title":"男子在工地宿舍睡觉时被捅身亡","ext":{"hotTags":0},"formatHotScore":"618万","heat_score":6184725,"hotScore":"6184725","hotTag":"0","img":"https:\/\/fyb-2.cdn.bcebos.com\/hotboard_image\/3d17a434baf2b0b31dc2f6bbd380b0ff","index":19,"linkurl":"https:\/\/www.baidu.com\/s?wd=%E7%94%B7%E5%AD%90%E5%9C%A8%E5%B7%A5%E5%9C%B0%E5%AE%BF%E8%88%8D%E7%9D%A1%E8%A7%89%E6%97%B6%E8%A2%AB%E6%8D%85%E8%BA%AB%E4%BA%A1&sa=fyb_n_homepage&rsv_dl=fyb_n_homepage&from=super&cl=3&tn=baidutop10&fr=top1000&rsv_idx=2&hisfilter=1","pre_tag":"","query":"男子在工地宿舍睡觉时被捅身亡","sa":"fyb_n_homepage","url":"https:\/\/www.baidu.com\/s?wd=%E7%94%B7%E5%AD%90%E5%9C%A8%E5%B7%A5%E5%9C%B0%E5%AE%BF%E8%88%8D%E7%9D%A1%E8%A7%89%E6%97%B6%E8%A2%AB%E6%8D%85%E8%BA%AB%E4%BA%A1&sa=fyb_n_homepage&rsv_dl=fyb_n_homepage&from=super","word":"男子在工地宿舍睡觉时被捅身亡"},{"card_title":"财政局副科长被举报出轨 纪委回应","ext":{"hotTags":0},"formatHotScore":"606万","heat_score":6066608,"hotScore":"6066608","hotTag":"0","img":"https:\/\/fyb-2.cdn.bcebos.com\/hotboard_image\/5e86d4c75ee54a06e9dec4ddd27db776","index":20,"linkurl":"https:\/\/www.baidu.com\/s?wd=%E8%B4%A2%E6%94%BF%E5%B1%80%E5%89%AF%E7%A7%91%E9%95%BF%E8%A2%AB%E4%B8%BE%E6%8A%A5%E5%87%BA%E8%BD%A8+%E7%BA%AA%E5%A7%94%E5%9B%9E%E5%BA%94&sa=fyb_n_homepage&rsv_dl=fyb_n_homepage&from=super&cl=3&tn=baidutop10&fr=top1000&rsv_idx=2&hisfilter=1","pre_tag":"","query":"财政局副科长被举报出轨 纪委回应","sa":"fyb_n_homepage","url":"https:\/\/www.baidu.com\/s?wd=%E8%B4%A2%E6%94%BF%E5%B1%80%E5%89%AF%E7%A7%91%E9%95%BF%E8%A2%AB%E4%B8%BE%E6%8A%A5%E5%87%BA%E8%BD%A8+%E7%BA%AA%E5%A7%94%E5%9B%9E%E5%BA%94&sa=fyb_n_homepage&rsv_dl=fyb_n_homepage&from=super","word":"财政局副科长被举报出轨 纪委回应"},{"card_title":"李晟被扒出曾背刺琼瑶黑历史","ext":{"hotTags":0},"formatHotScore":"599万","heat_score":5999232,"hotScore":"5999232","hotTag":"0","img":"https:\/\/fyb-2.cdn.bcebos.com\/hotboard_image\/84ff36ab1a3137b44ae23dbb95b2b69b","index":21,"linkurl":"https:\/\/www.baidu.com\/s?wd=%E6%9D%8E%E6%99%9F%E8%A2%AB%E6%89%92%E5%87%BA%E6%9B%BE%E8%83%8C%E5%88%BA%E7%90%BC%E7%91%B6%E9%BB%91%E5%8E%86%E5%8F%B2&sa=fyb_n_homepage&rsv_dl=fyb_n_homepage&from=super&cl=3&tn=baidutop10&fr=top1000&rsv_idx=2&hisfilter=1","pre_tag":"","query":"李晟被扒出曾背刺琼瑶黑历史","sa":"fyb_n_homepage","url":"https:\/\/www.baidu.com\/s?wd=%E6%9D%8E%E6%99%9F%E8%A2%AB%E6%89%92%E5%87%BA%E6%9B%BE%E8%83%8C%E5%88%BA%E7%90%BC%E7%91%B6%E9%BB%91%E5%8E%86%E5%8F%B2&sa=fyb_n_homepage&rsv_dl=fyb_n_homepage&from=super","word":"李晟被扒出曾背刺琼瑶黑历史"},{"card_title":"贝莱德“吞下”李嘉诚43个港口背后","ext":{"hotTags":0},"formatHotScore":"586万","heat_score":5862817,"hotScore":"5862817","hotTag":"0","img":"https:\/\/gips3.baidu.com\/it\/u=3805236591,2431602968&fm=3028&app=3028&f=PNG&fmt=auto&q=75&size=f352_234","index":22,"linkurl":"https:\/\/www.baidu.com\/s?wd=%E8%B4%9D%E8%8E%B1%E5%BE%B7%E2%80%9C%E5%90%9E%E4%B8%8B%E2%80%9D%E6%9D%8E%E5%98%89%E8%AF%9A43%E4%B8%AA%E6%B8%AF%E5%8F%A3%E8%83%8C%E5%90%8E&sa=fyb_n_homepage&rsv_dl=fyb_n_homepage&from=super&cl=3&tn=baidutop10&fr=top1000&rsv_idx=2&hisfilter=1","pre_tag":"","query":"贝莱德“吞下”李嘉诚43个港口背后","sa":"fyb_n_homepage","url":"https:\/\/www.baidu.com\/s?wd=%E8%B4%9D%E8%8E%B1%E5%BE%B7%E2%80%9C%E5%90%9E%E4%B8%8B%E2%80%9D%E6%9D%8E%E5%98%89%E8%AF%9A43%E4%B8%AA%E6%B8%AF%E5%8F%A3%E8%83%8C%E5%90%8E&sa=fyb_n_homepage&rsv_dl=fyb_n_homepage&from=super","word":"贝莱德“吞下”李嘉诚43个港口背后"},{"card_title":"媒体：加新总理是特朗普强硬对手","ext":{"hotTags":0},"formatHotScore":"578万","heat_score":5784810,"hotScore":"5784810","hotTag":"0","img":"https:\/\/gips3.baidu.com\/it\/u=3805236591,2431602968&fm=3028&app=3028&f=PNG&fmt=auto&q=75&size=f352_234","index":23,"linkurl":"https:\/\/www.baidu.com\/s?wd=%E5%AA%92%E4%BD%93%EF%BC%9A%E5%8A%A0%E6%96%B0%E6%80%BB%E7%90%86%E6%98%AF%E7%89%B9%E6%9C%97%E6%99%AE%E5%BC%BA%E7%A1%AC%E5%AF%B9%E6%89%8B&sa=fyb_n_homepage&rsv_dl=fyb_n_homepage&from=super&cl=3&tn=baidutop10&fr=top1000&rsv_idx=2&hisfilter=1","pre_tag":"","query":"媒体：加新总理是特朗普强硬对手","sa":"fyb_n_homepage","url":"https:\/\/www.baidu.com\/s?wd=%E5%AA%92%E4%BD%93%EF%BC%9A%E5%8A%A0%E6%96%B0%E6%80%BB%E7%90%86%E6%98%AF%E7%89%B9%E6%9C%97%E6%99%AE%E5%BC%BA%E7%A1%AC%E5%AF%B9%E6%89%8B&sa=fyb_n_homepage&rsv_dl=fyb_n_homepage&from=super","word":"媒体：加新总理是特朗普强硬对手"},{"card_title":"国防部：全时待战、随时能战、战之必胜","ext":{"hotTags":0},"formatHotScore":"566万","heat_score":5667340,"hotScore":"5667340","hotTag":"0","img":"https:\/\/fyb-2.cdn.bcebos.com\/hotboard_image\/ebf50c55b9d6457cf5413ac6754cc663","index":24,"linkurl":"https:\/\/www.baidu.com\/s?wd=%E5%9B%BD%E9%98%B2%E9%83%A8%EF%BC%9A%E5%85%A8%E6%97%B6%E5%BE%85%E6%88%98%E3%80%81%E9%9A%8F%E6%97%B6%E8%83%BD%E6%88%98%E3%80%81%E6%88%98%E4%B9%8B%E5%BF%85%E8%83%9C&sa=fyb_n_homepage&rsv_dl=fyb_n_homepage&from=super&cl=3&tn=baidutop10&fr=top1000&rsv_idx=2&hisfilter=1","pre_tag":"","query":"国防部：全时待战、随时能战、战之必胜","sa":"fyb_n_homepage","url":"https:\/\/www.baidu.com\/s?wd=%E5%9B%BD%E9%98%B2%E9%83%A8%EF%BC%9A%E5%85%A8%E6%97%B6%E5%BE%85%E6%88%98%E3%80%81%E9%9A%8F%E6%97%B6%E8%83%BD%E6%88%98%E3%80%81%E6%88%98%E4%B9%8B%E5%BF%85%E8%83%9C&sa=fyb_n_homepage&rsv_dl=fyb_n_homepage&from=super","word":"国防部：全时待战、随时能战、战之必胜"},{"card_title":"甲亢哥和大妈合影被美颜拍成外星人","ext":{"hotTags":3},"formatHotScore":"554万","heat_score":5547364,"hotScore":"5547364","hotTag":"3","img":"https:\/\/fyb-2.cdn.bcebos.com\/hotboard_image\/8373ab4a6dc744af80b45afbc65d7cb6","index":25,"linkurl":"https:\/\/www.baidu.com\/s?wd=%E7%94%B2%E4%BA%A2%E5%93%A5%E5%92%8C%E5%A4%A7%E5%A6%88%E5%90%88%E5%BD%B1%E8%A2%AB%E7%BE%8E%E9%A2%9C%E6%8B%8D%E6%88%90%E5%A4%96%E6%98%9F%E4%BA%BA&sa=fyb_n_homepage&rsv_dl=fyb_n_homepage&from=super&cl=3&tn=baidutop10&fr=top1000&rsv_idx=2&hisfilter=1","pre_tag":"","query":"甲亢哥和大妈合影被美颜拍成外星人","sa":"fyb_n_homepage","url":"https:\/\/www.baidu.com\/s?wd=%E7%94%B2%E4%BA%A2%E5%93%A5%E5%92%8C%E5%A4%A7%E5%A6%88%E5%90%88%E5%BD%B1%E8%A2%AB%E7%BE%8E%E9%A2%9C%E6%8B%8D%E6%88%90%E5%A4%96%E6%98%9F%E4%BA%BA&sa=fyb_n_homepage&rsv_dl=fyb_n_homepage&from=super","word":"甲亢哥和大妈合影被美颜拍成外星人"},{"card_title":"雷军富豪榜全球排名上升91位","ext":{"hotTags":0},"formatHotScore":"546万","heat_score":5463336,"hotScore":"5463336","hotTag":"0","img":"https:\/\/gips3.baidu.com\/it\/u=3805236591,2431602968&fm=3028&app=3028&f=PNG&fmt=auto&q=75&size=f352_234","index":26,"linkurl":"https:\/\/www.baidu.com\/s?wd=%E9%9B%B7%E5%86%9B%E5%AF%8C%E8%B1%AA%E6%A6%9C%E5%85%A8%E7%90%83%E6%8E%92%E5%90%8D%E4%B8%8A%E5%8D%8791%E4%BD%8D&sa=fyb_n_homepage&rsv_dl=fyb_n_homepage&from=super&cl=3&tn=baidutop10&fr=top1000&rsv_idx=2&hisfilter=1","pre_tag":"","query":"雷军富豪榜全球排名上升91位","sa":"fyb_n_homepage","url":"https:\/\/www.baidu.com\/s?wd=%E9%9B%B7%E5%86%9B%E5%AF%8C%E8%B1%AA%E6%A6%9C%E5%85%A8%E7%90%83%E6%8E%92%E5%90%8D%E4%B8%8A%E5%8D%8791%E4%BD%8D&sa=fyb_n_homepage&rsv_dl=fyb_n_homepage&from=super","word":"雷军富豪榜全球排名上升91位"},{"card_title":"42岁黄圣依亮相跑马拉松","ext":{"hotTags":0},"formatHotScore":"535万","heat_score":5354622,"hotScore":"5354622","hotTag":"0","img":"https:\/\/gips3.baidu.com\/it\/u=3805236591,2431602968&fm=3028&app=3028&f=PNG&fmt=auto&q=75&size=f352_234","index":27,"linkurl":"https:\/\/www.baidu.com\/s?wd=42%E5%B2%81%E9%BB%84%E5%9C%A3%E4%BE%9D%E4%BA%AE%E7%9B%B8%E8%B7%91%E9%A9%AC%E6%8B%89%E6%9D%BE&sa=fyb_n_homepage&rsv_dl=fyb_n_homepage&from=super&cl=3&tn=baidutop10&fr=top1000&rsv_idx=2&hisfilter=1","pre_tag":"","query":"42岁黄圣依亮相跑马拉松","sa":"fyb_n_homepage","url":"https:\/\/www.baidu.com\/s?wd=42%E5%B2%81%E9%BB%84%E5%9C%A3%E4%BE%9D%E4%BA%AE%E7%9B%B8%E8%B7%91%E9%A9%AC%E6%8B%89%E6%9D%BE&sa=fyb_n_homepage&rsv_dl=fyb_n_homepage&from=super","word":"42岁黄圣依亮相跑马拉松"},{"card_title":"宋茜回应没喝中药","ext":{"hotTags":0},"formatHotScore":"520万","heat_score":5207904,"hotScore":"5207904","hotTag":"0","img":"https:\/\/gips3.baidu.com\/it\/u=3805236591,2431602968&fm=3028&app=3028&f=PNG&fmt=auto&q=75&size=f352_234","index":28,"linkurl":"https:\/\/www.baidu.com\/s?wd=%E5%AE%8B%E8%8C%9C%E5%9B%9E%E5%BA%94%E6%B2%A1%E5%96%9D%E4%B8%AD%E8%8D%AF&sa=fyb_n_homepage&rsv_dl=fyb_n_homepage&from=super&cl=3&tn=baidutop10&fr=top1000&rsv_idx=2&hisfilter=1","pre_tag":"","query":"宋茜回应没喝中药","sa":"fyb_n_homepage","url":"https:\/\/www.baidu.com\/s?wd=%E5%AE%8B%E8%8C%9C%E5%9B%9E%E5%BA%94%E6%B2%A1%E5%96%9D%E4%B8%AD%E8%8D%AF&sa=fyb_n_homepage&rsv_dl=fyb_n_homepage&from=super","word":"宋茜回应没喝中药"},{"card_title":"商务部：中方已起诉美方加征关税措施","ext":{"hotTags":0},"formatHotScore":"511万","heat_score":5111454,"hotScore":"5111454","hotTag":"0","img":"https:\/\/fyb-2.cdn.bcebos.com\/hotboard_image\/7946f08929128d3546dc155144f5580e","index":29,"linkurl":"https:\/\/www.baidu.com\/s?wd=%E5%95%86%E5%8A%A1%E9%83%A8%EF%BC%9A%E4%B8%AD%E6%96%B9%E5%B7%B2%E8%B5%B7%E8%AF%89%E7%BE%8E%E6%96%B9%E5%8A%A0%E5%BE%81%E5%85%B3%E7%A8%8E%E6%8E%AA%E6%96%BD&sa=fyb_n_homepage&rsv_dl=fyb_n_homepage&from=super&cl=3&tn=baidutop10&fr=top1000&rsv_idx=2&hisfilter=1","pre_tag":"","query":"商务部：中方已起诉美方加征关税措施","sa":"fyb_n_homepage","url":"https:\/\/www.baidu.com\/s?wd=%E5%95%86%E5%8A%A1%E9%83%A8%EF%BC%9A%E4%B8%AD%E6%96%B9%E5%B7%B2%E8%B5%B7%E8%AF%89%E7%BE%8E%E6%96%B9%E5%8A%A0%E5%BE%81%E5%85%B3%E7%A8%8E%E6%8E%AA%E6%96%BD&sa=fyb_n_homepage&rsv_dl=fyb_n_homepage&from=super","word":"商务部：中方已起诉美方加征关税措施"},{"card_title":"歼-16女飞行员驾驶战机一飞冲天","ext":{"hotTags":0},"formatHotScore":"507万","heat_score":5072259,"hotScore":"5072259","hotTag":"0","img":"https:\/\/gips3.baidu.com\/it\/u=3805236591,2431602968&fm=3028&app=3028&f=PNG&fmt=auto&q=75&size=f352_234","index":30,"linkurl":"https:\/\/www.baidu.com\/s?wd=%E6%AD%BC-16%E5%A5%B3%E9%A3%9E%E8%A1%8C%E5%91%98%E9%A9%BE%E9%A9%B6%E6%88%98%E6%9C%BA%E4%B8%80%E9%A3%9E%E5%86%B2%E5%A4%A9&sa=fyb_n_homepage&rsv_dl=fyb_n_homepage&from=super&cl=3&tn=baidutop10&fr=top1000&rsv_idx=2&hisfilter=1","pre_tag":"","query":"歼-16女飞行员驾驶战机一飞冲天","sa":"fyb_n_homepage","url":"https:\/\/www.baidu.com\/s?wd=%E6%AD%BC-16%E5%A5%B3%E9%A3%9E%E8%A1%8C%E5%91%98%E9%A9%BE%E9%A9%B6%E6%88%98%E6%9C%BA%E4%B8%80%E9%A3%9E%E5%86%B2%E5%A4%A9&sa=fyb_n_homepage&rsv_dl=fyb_n_homepage&from=super","word":"歼-16女飞行员驾驶战机一飞冲天"},{"card_title":"赵露思因为躯体化记不住台词","ext":{"hotTags":0},"formatHotScore":"493万","heat_score":4939504,"hotScore":"4939504","hotTag":"0","img":"https:\/\/fyb-2.cdn.bcebos.com\/hotboard_image\/e7664ecf0ad362464f26cc7d5d2989c4","index":31,"linkurl":"https:\/\/www.baidu.com\/s?wd=%E8%B5%B5%E9%9C%B2%E6%80%9D%E5%9B%A0%E4%B8%BA%E8%BA%AF%E4%BD%93%E5%8C%96%E8%AE%B0%E4%B8%8D%E4%BD%8F%E5%8F%B0%E8%AF%8D&sa=fyb_n_homepage&rsv_dl=fyb_n_homepage&from=super&cl=3&tn=baidutop10&fr=top1000&rsv_idx=2&hisfilter=1","pre_tag":"","query":"赵露思因为躯体化记不住台词","sa":"fyb_n_homepage","url":"https:\/\/www.baidu.com\/s?wd=%E8%B5%B5%E9%9C%B2%E6%80%9D%E5%9B%A0%E4%B8%BA%E8%BA%AF%E4%BD%93%E5%8C%96%E8%AE%B0%E4%B8%8D%E4%BD%8F%E5%8F%B0%E8%AF%8D&sa=fyb_n_homepage&rsv_dl=fyb_n_homepage&from=super","word":"赵露思因为躯体化记不住台词"},{"card_title":"年薪百万副总裁伪造公章被立案","ext":{"hotTags":0},"formatHotScore":"485万","heat_score":4850020,"hotScore":"4850020","hotTag":"0","img":"https:\/\/gips3.baidu.com\/it\/u=3805236591,2431602968&fm=3028&app=3028&f=PNG&fmt=auto&q=75&size=f352_234","index":32,"linkurl":"https:\/\/www.baidu.com\/s?wd=%E5%B9%B4%E8%96%AA%E7%99%BE%E4%B8%87%E5%89%AF%E6%80%BB%E8%A3%81%E4%BC%AA%E9%80%A0%E5%85%AC%E7%AB%A0%E8%A2%AB%E7%AB%8B%E6%A1%88&sa=fyb_n_homepage&rsv_dl=fyb_n_homepage&from=super&cl=3&tn=baidutop10&fr=top1000&rsv_idx=2&hisfilter=1","pre_tag":"","query":"年薪百万副总裁伪造公章被立案","sa":"fyb_n_homepage","url":"https:\/\/www.baidu.com\/s?wd=%E5%B9%B4%E8%96%AA%E7%99%BE%E4%B8%87%E5%89%AF%E6%80%BB%E8%A3%81%E4%BC%AA%E9%80%A0%E5%85%AC%E7%AB%A0%E8%A2%AB%E7%AB%8B%E6%A1%88&sa=fyb_n_homepage&rsv_dl=fyb_n_homepage&from=super","word":"年薪百万副总裁伪造公章被立案"},{"card_title":"69岁阿姨自驾环游中国走出抑郁","ext":{"hotTags":0},"formatHotScore":"475万","heat_score":4756465,"hotScore":"4756465","hotTag":"0","img":"https:\/\/gips3.baidu.com\/it\/u=3805236591,2431602968&fm=3028&app=3028&f=PNG&fmt=auto&q=75&size=f352_234","index":33,"linkurl":"https:\/\/www.baidu.com\/s?wd=69%E5%B2%81%E9%98%BF%E5%A7%A8%E8%87%AA%E9%A9%BE%E7%8E%AF%E6%B8%B8%E4%B8%AD%E5%9B%BD%E8%B5%B0%E5%87%BA%E6%8A%91%E9%83%81&sa=fyb_n_homepage&rsv_dl=fyb_n_homepage&from=super&cl=3&tn=baidutop10&fr=top1000&rsv_idx=2&hisfilter=1","pre_tag":"","query":"69岁阿姨自驾环游中国走出抑郁","sa":"fyb_n_homepage","url":"https:\/\/www.baidu.com\/s?wd=69%E5%B2%81%E9%98%BF%E5%A7%A8%E8%87%AA%E9%A9%BE%E7%8E%AF%E6%B8%B8%E4%B8%AD%E5%9B%BD%E8%B5%B0%E5%87%BA%E6%8A%91%E9%83%81&sa=fyb_n_homepage&rsv_dl=fyb_n_homepage&from=super","word":"69岁阿姨自驾环游中国走出抑郁"},{"card_title":"刘学义《念无双》完全就是“男妲己”","ext":{"hotTags":0},"formatHotScore":"460万","heat_score":4602752,"hotScore":"4602752","hotTag":"0","img":"https:\/\/gips3.baidu.com\/it\/u=3805236591,2431602968&fm=3028&app=3028&f=PNG&fmt=auto&q=75&size=f352_234","index":34,"linkurl":"https:\/\/www.baidu.com\/s?wd=%E5%88%98%E5%AD%A6%E4%B9%89%E3%80%8A%E5%BF%B5%E6%97%A0%E5%8F%8C%E3%80%8B%E5%AE%8C%E5%85%A8%E5%B0%B1%E6%98%AF%E2%80%9C%E7%94%B7%E5%A6%B2%E5%B7%B1%E2%80%9D&sa=fyb_n_homepage&rsv_dl=fyb_n_homepage&from=super&cl=3&tn=baidutop10&fr=top1000&rsv_idx=2&hisfilter=1","pre_tag":"","query":"刘学义《念无双》完全就是“男妲己”","sa":"fyb_n_homepage","url":"https:\/\/www.baidu.com\/s?wd=%E5%88%98%E5%AD%A6%E4%B9%89%E3%80%8A%E5%BF%B5%E6%97%A0%E5%8F%8C%E3%80%8B%E5%AE%8C%E5%85%A8%E5%B0%B1%E6%98%AF%E2%80%9C%E7%94%B7%E5%A6%B2%E5%B7%B1%E2%80%9D&sa=fyb_n_homepage&rsv_dl=fyb_n_homepage&from=super","word":"刘学义《念无双》完全就是“男妲己”"},{"card_title":"国防部：G7不要总是想着给人当教师爷","ext":{"hotTags":0},"formatHotScore":"452万","heat_score":4529476,"hotScore":"4529476","hotTag":"0","img":"https:\/\/fyb-2.cdn.bcebos.com\/hotboard_image\/fc0681d53c3b61f1d1e3ea16d2714f1c","index":35,"linkurl":"https:\/\/www.baidu.com\/s?wd=%E5%9B%BD%E9%98%B2%E9%83%A8%EF%BC%9AG7%E4%B8%8D%E8%A6%81%E6%80%BB%E6%98%AF%E6%83%B3%E7%9D%80%E7%BB%99%E4%BA%BA%E5%BD%93%E6%95%99%E5%B8%88%E7%88%B7&sa=fyb_n_homepage&rsv_dl=fyb_n_homepage&from=super&cl=3&tn=baidutop10&fr=top1000&rsv_idx=2&hisfilter=1","pre_tag":"","query":"国防部：G7不要总是想着给人当教师爷","sa":"fyb_n_homepage","url":"https:\/\/www.baidu.com\/s?wd=%E5%9B%BD%E9%98%B2%E9%83%A8%EF%BC%9AG7%E4%B8%8D%E8%A6%81%E6%80%BB%E6%98%AF%E6%83%B3%E7%9D%80%E7%BB%99%E4%BA%BA%E5%BD%93%E6%95%99%E5%B8%88%E7%88%B7&sa=fyb_n_homepage&rsv_dl=fyb_n_homepage&from=super","word":"国防部：G7不要总是想着给人当教师爷"},{"card_title":"第一批挣外国人钱的年轻人已不上班","ext":{"hotTags":0},"formatHotScore":"449万","heat_score":4493129,"hotScore":"4493129","hotTag":"0","img":"https:\/\/gips3.baidu.com\/it\/u=3805236591,2431602968&fm=3028&app=3028&f=PNG&fmt=auto&q=75&size=f352_234","index":36,"linkurl":"https:\/\/www.baidu.com\/s?wd=%E7%AC%AC%E4%B8%80%E6%89%B9%E6%8C%A3%E5%A4%96%E5%9B%BD%E4%BA%BA%E9%92%B1%E7%9A%84%E5%B9%B4%E8%BD%BB%E4%BA%BA%E5%B7%B2%E4%B8%8D%E4%B8%8A%E7%8F%AD&sa=fyb_n_homepage&rsv_dl=fyb_n_homepage&from=super&cl=3&tn=baidutop10&fr=top1000&rsv_idx=2&hisfilter=1","pre_tag":"","query":"第一批挣外国人钱的年轻人已不上班","sa":"fyb_n_homepage","url":"https:\/\/www.baidu.com\/s?wd=%E7%AC%AC%E4%B8%80%E6%89%B9%E6%8C%A3%E5%A4%96%E5%9B%BD%E4%BA%BA%E9%92%B1%E7%9A%84%E5%B9%B4%E8%BD%BB%E4%BA%BA%E5%B7%B2%E4%B8%8D%E4%B8%8A%E7%8F%AD&sa=fyb_n_homepage&rsv_dl=fyb_n_homepage&from=super","word":"第一批挣外国人钱的年轻人已不上班"},{"card_title":"夫妻开店遭房东刁难亏本后续","ext":{"hotTags":0},"formatHotScore":"434万","heat_score":4344058,"hotScore":"4344058","hotTag":"0","img":"https:\/\/gips3.baidu.com\/it\/u=3805236591,2431602968&fm=3028&app=3028&f=PNG&fmt=auto&q=75&size=f352_234","index":37,"linkurl":"https:\/\/www.baidu.com\/s?wd=%E5%A4%AB%E5%A6%BB%E5%BC%80%E5%BA%97%E9%81%AD%E6%88%BF%E4%B8%9C%E5%88%81%E9%9A%BE%E4%BA%8F%E6%9C%AC%E5%90%8E%E7%BB%AD&sa=fyb_n_homepage&rsv_dl=fyb_n_homepage&from=super&cl=3&tn=baidutop10&fr=top1000&rsv_idx=2&hisfilter=1","pre_tag":"","query":"夫妻开店遭房东刁难亏本后续","sa":"fyb_n_homepage","url":"https:\/\/www.baidu.com\/s?wd=%E5%A4%AB%E5%A6%BB%E5%BC%80%E5%BA%97%E9%81%AD%E6%88%BF%E4%B8%9C%E5%88%81%E9%9A%BE%E4%BA%8F%E6%9C%AC%E5%90%8E%E7%BB%AD&sa=fyb_n_homepage&rsv_dl=fyb_n_homepage&from=super","word":"夫妻开店遭房东刁难亏本后续"},{"card_title":"张雨霏恢复训练像身体被掏空","ext":{"hotTags":0},"formatHotScore":"423万","heat_score":4235422,"hotScore":"4235422","hotTag":"0","img":"https:\/\/gips3.baidu.com\/it\/u=3805236591,2431602968&fm=3028&app=3028&f=PNG&fmt=auto&q=75&size=f352_234","index":38,"linkurl":"https:\/\/www.baidu.com\/s?wd=%E5%BC%A0%E9%9B%A8%E9%9C%8F%E6%81%A2%E5%A4%8D%E8%AE%AD%E7%BB%83%E5%83%8F%E8%BA%AB%E4%BD%93%E8%A2%AB%E6%8E%8F%E7%A9%BA&sa=fyb_n_homepage&rsv_dl=fyb_n_homepage&from=super&cl=3&tn=baidutop10&fr=top1000&rsv_idx=2&hisfilter=1","pre_tag":"","query":"张雨霏恢复训练像身体被掏空","sa":"fyb_n_homepage","url":"https:\/\/www.baidu.com\/s?wd=%E5%BC%A0%E9%9B%A8%E9%9C%8F%E6%81%A2%E5%A4%8D%E8%AE%AD%E7%BB%83%E5%83%8F%E8%BA%AB%E4%BD%93%E8%A2%AB%E6%8E%8F%E7%A9%BA&sa=fyb_n_homepage&rsv_dl=fyb_n_homepage&from=super","word":"张雨霏恢复训练像身体被掏空"},{"card_title":"女子减肥上瘾身高1米6体重仅52斤","ext":{"hotTags":0},"formatHotScore":"412万","heat_score":4127767,"hotScore":"4127767","hotTag":"0","img":"https:\/\/fyb-2.cdn.bcebos.com\/hotboard_image\/be455c2698c5667cb9ae538cedb64af7","index":39,"linkurl":"https:\/\/www.baidu.com\/s?wd=%E5%A5%B3%E5%AD%90%E5%87%8F%E8%82%A5%E4%B8%8A%E7%98%BE%E8%BA%AB%E9%AB%981%E7%B1%B36%E4%BD%93%E9%87%8D%E4%BB%8552%E6%96%A4&sa=fyb_n_homepage&rsv_dl=fyb_n_homepage&from=super&cl=3&tn=baidutop10&fr=top1000&rsv_idx=2&hisfilter=1","pre_tag":"","query":"女子减肥上瘾身高1米6体重仅52斤","sa":"fyb_n_homepage","url":"https:\/\/www.baidu.com\/s?wd=%E5%A5%B3%E5%AD%90%E5%87%8F%E8%82%A5%E4%B8%8A%E7%98%BE%E8%BA%AB%E9%AB%981%E7%B1%B36%E4%BD%93%E9%87%8D%E4%BB%8552%E6%96%A4&sa=fyb_n_homepage&rsv_dl=fyb_n_homepage&from=super","word":"女子减肥上瘾身高1米6体重仅52斤"},{"card_title":"特朗普：对所有进口汽车征收25%关税","ext":{"hotTags":3},"formatHotScore":"408万","heat_score":4085298,"hotScore":"4085298","hotTag":"3","img":"https:\/\/fyb-2.cdn.bcebos.com\/hotboard_image\/de1c83d9623cf7a69443372e5ca0a88b","index":40,"linkurl":"https:\/\/www.baidu.com\/s?wd=%E7%89%B9%E6%9C%97%E6%99%AE%EF%BC%9A%E5%AF%B9%E6%89%80%E6%9C%89%E8%BF%9B%E5%8F%A3%E6%B1%BD%E8%BD%A6%E5%BE%81%E6%94%B625%25%E5%85%B3%E7%A8%8E&sa=fyb_n_homepage&rsv_dl=fyb_n_homepage&from=super&cl=3&tn=baidutop10&fr=top1000&rsv_idx=2&hisfilter=1","pre_tag":"","query":"特朗普：对所有进口汽车征收25%关税","sa":"fyb_n_homepage","url":"https:\/\/www.baidu.com\/s?wd=%E7%89%B9%E6%9C%97%E6%99%AE%EF%BC%9A%E5%AF%B9%E6%89%80%E6%9C%89%E8%BF%9B%E5%8F%A3%E6%B1%BD%E8%BD%A6%E5%BE%81%E6%94%B625%25%E5%85%B3%E7%A8%8E&sa=fyb_n_homepage&rsv_dl=fyb_n_homepage&from=super","word":"特朗普：对所有进口汽车征收25%关税"},{"card_title":"卡尼耍狠：无法与中国共享贸易价值观","ext":{"hotTags":0},"formatHotScore":"393万","heat_score":3937018,"hotScore":"3937018","hotTag":"0","img":"https:\/\/fyb-2.cdn.bcebos.com\/hotboard_image\/4f81b78a96311f6480d5861f7efdd876","index":41,"linkurl":"https:\/\/www.baidu.com\/s?wd=%E5%8D%A1%E5%B0%BC%E8%80%8D%E7%8B%A0%EF%BC%9A%E6%97%A0%E6%B3%95%E4%B8%8E%E4%B8%AD%E5%9B%BD%E5%85%B1%E4%BA%AB%E8%B4%B8%E6%98%93%E4%BB%B7%E5%80%BC%E8%A7%82&sa=fyb_n_homepage&rsv_dl=fyb_n_homepage&from=super&cl=3&tn=baidutop10&fr=top1000&rsv_idx=2&hisfilter=1","pre_tag":"","query":"卡尼耍狠：无法与中国共享贸易价值观","sa":"fyb_n_homepage","url":"https:\/\/www.baidu.com\/s?wd=%E5%8D%A1%E5%B0%BC%E8%80%8D%E7%8B%A0%EF%BC%9A%E6%97%A0%E6%B3%95%E4%B8%8E%E4%B8%AD%E5%9B%BD%E5%85%B1%E4%BA%AB%E8%B4%B8%E6%98%93%E4%BB%B7%E5%80%BC%E8%A7%82&sa=fyb_n_homepage&rsv_dl=fyb_n_homepage&from=super","word":"卡尼耍狠：无法与中国共享贸易价值观"},{"card_title":"男子走进寺庙 把香灰倒出来观察香炉","ext":{"hotTags":0},"formatHotScore":"382万","heat_score":3827745,"hotScore":"3827745","hotTag":"0","img":"https:\/\/gips3.baidu.com\/it\/u=3805236591,2431602968&fm=3028&app=3028&f=PNG&fmt=auto&q=75&size=f352_234","index":42,"linkurl":"https:\/\/www.baidu.com\/s?wd=%E7%94%B7%E5%AD%90%E8%B5%B0%E8%BF%9B%E5%AF%BA%E5%BA%99+%E6%8A%8A%E9%A6%99%E7%81%B0%E5%80%92%E5%87%BA%E6%9D%A5%E8%A7%82%E5%AF%9F%E9%A6%99%E7%82%89&sa=fyb_n_homepage&rsv_dl=fyb_n_homepage&from=super&cl=3&tn=baidutop10&fr=top1000&rsv_idx=2&hisfilter=1","pre_tag":"","query":"男子走进寺庙 把香灰倒出来观察香炉","sa":"fyb_n_homepage","url":"https:\/\/www.baidu.com\/s?wd=%E7%94%B7%E5%AD%90%E8%B5%B0%E8%BF%9B%E5%AF%BA%E5%BA%99+%E6%8A%8A%E9%A6%99%E7%81%B0%E5%80%92%E5%87%BA%E6%9D%A5%E8%A7%82%E5%AF%9F%E9%A6%99%E7%82%89&sa=fyb_n_homepage&rsv_dl=fyb_n_homepage&from=super","word":"男子走进寺庙 把香灰倒出来观察香炉"},{"card_title":"2小时车程变2分钟是什么体验","ext":{"hotTags":0},"formatHotScore":"378万","heat_score":3785792,"hotScore":"3785792","hotTag":"0","img":"https:\/\/gips3.baidu.com\/it\/u=3805236591,2431602968&fm=3028&app=3028&f=PNG&fmt=auto&q=75&size=f352_234","index":43,"linkurl":"https:\/\/www.baidu.com\/s?wd=2%E5%B0%8F%E6%97%B6%E8%BD%A6%E7%A8%8B%E5%8F%982%E5%88%86%E9%92%9F%E6%98%AF%E4%BB%80%E4%B9%88%E4%BD%93%E9%AA%8C&sa=fyb_n_homepage&rsv_dl=fyb_n_homepage&from=super&cl=3&tn=baidutop10&fr=top1000&rsv_idx=2&hisfilter=1","pre_tag":"","query":"2小时车程变2分钟是什么体验","sa":"fyb_n_homepage","url":"https:\/\/www.baidu.com\/s?wd=2%E5%B0%8F%E6%97%B6%E8%BD%A6%E7%A8%8B%E5%8F%982%E5%88%86%E9%92%9F%E6%98%AF%E4%BB%80%E4%B9%88%E4%BD%93%E9%AA%8C&sa=fyb_n_homepage&rsv_dl=fyb_n_homepage&from=super","word":"2小时车程变2分钟是什么体验"},{"card_title":"966万元大奖过期前被“压哨”领走","ext":{"hotTags":0},"formatHotScore":"367万","heat_score":3673285,"hotScore":"3673285","hotTag":"0","img":"https:\/\/fyb-2.cdn.bcebos.com\/hotboard_image\/8424a23f51c72dd673b939923232999d","index":44,"linkurl":"https:\/\/www.baidu.com\/s?wd=966%E4%B8%87%E5%85%83%E5%A4%A7%E5%A5%96%E8%BF%87%E6%9C%9F%E5%89%8D%E8%A2%AB%E2%80%9C%E5%8E%8B%E5%93%A8%E2%80%9D%E9%A2%86%E8%B5%B0&sa=fyb_n_homepage&rsv_dl=fyb_n_homepage&from=super&cl=3&tn=baidutop10&fr=top1000&rsv_idx=2&hisfilter=1","pre_tag":"","query":"966万元大奖过期前被“压哨”领走","sa":"fyb_n_homepage","url":"https:\/\/www.baidu.com\/s?wd=966%E4%B8%87%E5%85%83%E5%A4%A7%E5%A5%96%E8%BF%87%E6%9C%9F%E5%89%8D%E8%A2%AB%E2%80%9C%E5%8E%8B%E5%93%A8%E2%80%9D%E9%A2%86%E8%B5%B0&sa=fyb_n_homepage&rsv_dl=fyb_n_homepage&from=super","word":"966万元大奖过期前被“压哨”领走"},{"card_title":"高二女生用扭扭棒复刻明凤冠","ext":{"hotTags":0},"formatHotScore":"359万","heat_score":3596768,"hotScore":"3596768","hotTag":"0","img":"https:\/\/gips3.baidu.com\/it\/u=3805236591,2431602968&fm=3028&app=3028&f=PNG&fmt=auto&q=75&size=f352_234","index":45,"linkurl":"https:\/\/www.baidu.com\/s?wd=%E9%AB%98%E4%BA%8C%E5%A5%B3%E7%94%9F%E7%94%A8%E6%89%AD%E6%89%AD%E6%A3%92%E5%A4%8D%E5%88%BB%E6%98%8E%E5%87%A4%E5%86%A0&sa=fyb_n_homepage&rsv_dl=fyb_n_homepage&from=super&cl=3&tn=baidutop10&fr=top1000&rsv_idx=2&hisfilter=1","pre_tag":"","query":"高二女生用扭扭棒复刻明凤冠","sa":"fyb_n_homepage","url":"https:\/\/www.baidu.com\/s?wd=%E9%AB%98%E4%BA%8C%E5%A5%B3%E7%94%9F%E7%94%A8%E6%89%AD%E6%89%AD%E6%A3%92%E5%A4%8D%E5%88%BB%E6%98%8E%E5%87%A4%E5%86%A0&sa=fyb_n_homepage&rsv_dl=fyb_n_homepage&from=super","word":"高二女生用扭扭棒复刻明凤冠"},{"card_title":"小鹏汽车回应屏幕夹断儿童手指事件","ext":{"hotTags":0},"formatHotScore":"347万","heat_score":3471502,"hotScore":"3471502","hotTag":"0","img":"https:\/\/gips3.baidu.com\/it\/u=3805236591,2431602968&fm=3028&app=3028&f=PNG&fmt=auto&q=75&size=f352_234","index":46,"linkurl":"https:\/\/www.baidu.com\/s?wd=%E5%B0%8F%E9%B9%8F%E6%B1%BD%E8%BD%A6%E5%9B%9E%E5%BA%94%E5%B1%8F%E5%B9%95%E5%A4%B9%E6%96%AD%E5%84%BF%E7%AB%A5%E6%89%8B%E6%8C%87%E4%BA%8B%E4%BB%B6&sa=fyb_n_homepage&rsv_dl=fyb_n_homepage&from=super&cl=3&tn=baidutop10&fr=top1000&rsv_idx=2&hisfilter=1","pre_tag":"","query":"小鹏汽车回应屏幕夹断儿童手指事件","sa":"fyb_n_homepage","url":"https:\/\/www.baidu.com\/s?wd=%E5%B0%8F%E9%B9%8F%E6%B1%BD%E8%BD%A6%E5%9B%9E%E5%BA%94%E5%B1%8F%E5%B9%95%E5%A4%B9%E6%96%AD%E5%84%BF%E7%AB%A5%E6%89%8B%E6%8C%87%E4%BA%8B%E4%BB%B6&sa=fyb_n_homepage&rsv_dl=fyb_n_homepage&from=super","word":"小鹏汽车回应屏幕夹断儿童手指事件"},{"card_title":"恒大汽车回应股价一度暴涨超230%","ext":{"hotTags":0},"formatHotScore":"337万","heat_score":3376632,"hotScore":"3376632","hotTag":"0","img":"https:\/\/gips3.baidu.com\/it\/u=3805236591,2431602968&fm=3028&app=3028&f=PNG&fmt=auto&q=75&size=f352_234","index":47,"linkurl":"https:\/\/www.baidu.com\/s?wd=%E6%81%92%E5%A4%A7%E6%B1%BD%E8%BD%A6%E5%9B%9E%E5%BA%94%E8%82%A1%E4%BB%B7%E4%B8%80%E5%BA%A6%E6%9A%B4%E6%B6%A8%E8%B6%85230%25&sa=fyb_n_homepage&rsv_dl=fyb_n_homepage&from=super&cl=3&tn=baidutop10&fr=top1000&rsv_idx=2&hisfilter=1","pre_tag":"","query":"恒大汽车回应股价一度暴涨超230%","sa":"fyb_n_homepage","url":"https:\/\/www.baidu.com\/s?wd=%E6%81%92%E5%A4%A7%E6%B1%BD%E8%BD%A6%E5%9B%9E%E5%BA%94%E8%82%A1%E4%BB%B7%E4%B8%80%E5%BA%A6%E6%9A%B4%E6%B6%A8%E8%B6%85230%25&sa=fyb_n_homepage&rsv_dl=fyb_n_homepage&from=super","word":"恒大汽车回应股价一度暴涨超230%"},{"card_title":"南京高科一项目“割肉”促销","ext":{"hotTags":0},"formatHotScore":"321万","heat_score":3211842,"hotScore":"3211842","hotTag":"0","img":"https:\/\/gips3.baidu.com\/it\/u=3805236591,2431602968&fm=3028&app=3028&f=PNG&fmt=auto&q=75&size=f352_234","index":48,"linkurl":"https:\/\/www.baidu.com\/s?wd=%E5%8D%97%E4%BA%AC%E9%AB%98%E7%A7%91%E4%B8%80%E9%A1%B9%E7%9B%AE%E2%80%9C%E5%89%B2%E8%82%89%E2%80%9D%E4%BF%83%E9%94%80&sa=fyb_n_homepage&rsv_dl=fyb_n_homepage&from=super&cl=3&tn=baidutop10&fr=top1000&rsv_idx=2&hisfilter=1","pre_tag":"","query":"南京高科一项目“割肉”促销","sa":"fyb_n_homepage","url":"https:\/\/www.baidu.com\/s?wd=%E5%8D%97%E4%BA%AC%E9%AB%98%E7%A7%91%E4%B8%80%E9%A1%B9%E7%9B%AE%E2%80%9C%E5%89%B2%E8%82%89%E2%80%9D%E4%BF%83%E9%94%80&sa=fyb_n_homepage&rsv_dl=fyb_n_homepage&from=super","word":"南京高科一项目“割肉”促销"},{"card_title":"小伙靠书法帮父母还清2000万巨债","ext":{"hotTags":3},"formatHotScore":"312万","heat_score":3124946,"hotScore":"3124946","hotTag":"3","img":"https:\/\/fyb-2.cdn.bcebos.com\/hotboard_image\/5a62b61a07c2f024fde92c79385a1ddb","index":49,"linkurl":"https:\/\/www.baidu.com\/s?wd=%E5%B0%8F%E4%BC%99%E9%9D%A0%E4%B9%A6%E6%B3%95%E5%B8%AE%E7%88%B6%E6%AF%8D%E8%BF%98%E6%B8%852000%E4%B8%87%E5%B7%A8%E5%80%BA&sa=fyb_n_homepage&rsv_dl=fyb_n_homepage&from=super&cl=3&tn=baidutop10&fr=top1000&rsv_idx=2&hisfilter=1","pre_tag":"","query":"小伙靠书法帮父母还清2000万巨债","sa":"fyb_n_homepage","url":"https:\/\/www.baidu.com\/s?wd=%E5%B0%8F%E4%BC%99%E9%9D%A0%E4%B9%A6%E6%B3%95%E5%B8%AE%E7%88%B6%E6%AF%8D%E8%BF%98%E6%B8%852000%E4%B8%87%E5%B7%A8%E5%80%BA&sa=fyb_n_homepage&rsv_dl=fyb_n_homepage&from=super","word":"小伙靠书法帮父母还清2000万巨债"}],"tab":"realtime","tabName":"热搜榜","url":"https:\/\/top.baidu.com\/board"}]}</script><script type="text/javascript">"use strict";function _classCallCheck(instance,Constructor){if(!(instance instanceof Constructor)){throw new TypeError("Cannot call a class as a function")}}function _defineProperties(target,props){for(var i=0;i<props.length;i++){var descriptor=props[i];descriptor.enumerable=descriptor.enumerable||false;descriptor.configurable=true;if("value"in descriptor)descriptor.writable=true;Object.defineProperty(target,descriptor.key,descriptor)}}function _createClass(Constructor,protoProps,staticProps){
if(protoProps)_defineProperties(Constructor.prototype,protoProps);if(staticProps)_defineProperties(Constructor,staticProps);return Constructor}function _defineProperty(obj,key,value){if(key in obj){Object.defineProperty(obj,key,{value:value,enumerable:true,configurable:true,writable:true})}else{obj[key]=value}return obj}(function(){var config={sample:{jsnotfound:.02,imgError:.02,jserror:.02},logServer:"https://sp1.baidu.com/5b1ZeDe5KgQFm2e88IuM_a/mwb2.gif"};var imgErrorLen=0
;function alwaysMonitor(info){if(bds.comm.alwaysMonitor==="1"){info.alwaysMonitor=1;return true}return false}var LocalCache=function(){function LocalCache(){_classCallCheck(this,LocalCache);_defineProperty(this,"lsName","pcSpyLocalCache");_defineProperty(this,"tmpList",[]);_defineProperty(this,"timer",null)}_createClass(LocalCache,[{key:"addLog",value:function addLog(info){var _this=this;var infoStr=JSON.stringify(info);this.tmpList.push(infoStr);if(this.timer){clearTimeout(this.timer)}
this.timer=setTimeout(function(){_this.save()},500)}},{key:"getData",value:function getData(cb){try{var storageStr=localStorage.getItem(this.lsName);if(storageStr){cb(storageStr)}else{cb("")}}catch(e){console.error(e);cb("")}}},{key:"save",value:function save(){var _this2=this;this.getData(function(str){var content=str+"\n"+_this2.tmpList.join("\n");try{localStorage.setItem(_this2.lsName,content)}catch(error){}})}}],[{key:"getInstance",value:function getInstance(){if(!this.instance){
this.instance=new LocalCache}return this.instance}}]);return LocalCache}();_defineProperty(LocalCache,"instance",void 0);function shouldSend(sample){if(!sample){return false}var hitCookie=document.cookie.indexOf("webbtest=1")>-1;return hitCookie||Math.random()<sample}function send(obj){var type=arguments.length>1&&arguments[1]!==undefined?arguments[1]:"except";var lid=bds.comm.qid||bds.comm.queryId;var time=Date.now();var spyCache=LocalCache.getInstance();spyCache.addLog({from:"result",
info:obj.info,lid:lid,group:obj.group,ts:time,url:location.href});if(!shouldSend(config.sample[obj.group])&&!alwaysMonitor(obj.info)){return""}var pid="1_87";if(bds.comm.ishome){pid="1_79"}var logUrl=config.logServer+"?pid="+pid+"&lid="+lid+"&ts="+time+"&type="+type+"&group="+obj.group+"&info="+encodeURIComponent(JSON.stringify(obj.info))+"&dim="+encodeURIComponent(JSON.stringify(obj.dim||{}));var img=new Image;img.src=logUrl;return logUrl}function filterError(msg,stack){if(stack.indexOf(
"chrome-extension://")>-1||stack.indexOf("moz-extension://")>-1){return false}return true}function jsError(event,info){var obj={info:info,dim:{},group:"jserror"};var error=event.error||{};var errorStack=error.stack||"";if(event.message&&filterError(event.message,errorStack)){obj.info.msg=event.message;obj.info.file=event.filename;obj.info.ln=event.lineno;obj.info.col=event.colno;obj.info.stack=errorStack.split("\n").slice(0,3).join("\n");send(obj)}}function getPathTo(element){
var _element$parentNode;if(element.id!==""){return'id("'.concat(element.id,'")')}if(element===document.body){return element.tagName}var ix=0;var siblings=(element===null||element===void 0?void 0:(_element$parentNode=element.parentNode)===null||_element$parentNode===void 0?void 0:_element$parentNode.childNodes)?element.parentNode.childNodes:[];for(var i=0;i<siblings.length;i++){var sibling=siblings[i];if(sibling===element){return getPathTo(sibling.parentNode)+"/"+element.tagName+"["+(ix+1)+"]"}
if(sibling.nodeType===1&&sibling.tagName===element.tagName){ix++}}return""}function getLocation(href){var ele=document.createElement("a");ele.href=href;return ele}function getUrlHost(url){var host=getLocation(url).hostname;if(!/\.baidu|bcebos|bdstatic|baidubce|bdimg\.com/.test(host)){return"other"}return host}function imgError(target,info){var imgSrc=target.getAttribute("src")||"";info.msg=imgSrc;info.errLen=++imgErrorLen;info.xpath=getPathTo(target);send({info:info,dim:{host:getUrlHost(imgSrc)
},group:"imgError"},"et_comm")}function sourceError(event,from){try{var obj={info:{},dim:{},group:""};var info=obj.info;var target=event.target;var dataConnection=navigator.connection||{};info.downlink=dataConnection.downlink;info.effectiveType=dataConnection.effectiveType;info.rtt=dataConnection.rtt;info.deviceMemory=navigator.deviceMemory||0;info.hardwareConcurrency=navigator.hardwareConcurrency||0;info.saveData=!!dataConnection.saveData;if(target&&target.nodeName&&target.nodeName.toLowerCase(
)==="img"){imgError(target,info);return}var localName=target.localName||"";var errorLink=target.src||"";if(localName&&localName==="script"){obj.group="jsnotfound";info.msg=errorLink;info.file=errorLink;send(obj);return}if(target===window||from==="onerror"){jsError(event,info)}}catch(e){}}window.addEventListener&&window.addEventListener("error",sourceError,true)})();</script><script>
this.globalThis || (this.globalThis = this);
</script><noscript><meta http-equiv="refresh" content="0; url=http://www.baidu.com/baidu.html?from=noscript" /></noscript></head><body class="cos-pc 
  s-manhattan-index" ssr="n"><script>
    if (navigator.userAgent.indexOf('Edge') > -1) {
        var body = document.querySelector('body');
        body.className += ' browser-edge';
    }
</script><textarea id="s_is_result_css" style="display:none;">
&lt;style data-for=&quot;result&quot; type=&quot;text/css&quot; &gt;html{font-size:100px}html body{font-size:.14rem;font-size:14px}body{color:#333;background:#fff;padding:6px 0 0;margin:0;position:relative}body,th,td,.p1,.p2{font-family:arial}p,form,ol,ul,li,dl,dt,dd,h3{margin:0;padding:0;list-style:none}input{padding-top:0;padding-bottom:0;-moz-box-sizing:border-box;-webkit-box-sizing:border-box;box-sizing:border-box}table,img{border:0}td{font-size:9pt;line-height:18px}em{font-style:normal}em{font-style:normal;color:#c00}a em{text-decoration:underline}cite{font-style:normal;color:green}.m,a.m{color:#666}a.m:visited{color:#606}.g,a.g{color:green}.c{color:#77c}.f14{font-size:14px}.f10{font-size:10.5pt}.f16{font-size:16px}.f13{font-size:13px}.bg{background-image:url(https://pss.bdstatic.com/r/www/cache/static/protocol/https/global/img/icons_441e82f.png);_background-image:url(https://pss.bdstatic.com/r/www/cache/static/protocol/https/global/img/icons_d5b04cc.gif);background-repeat:no-repeat}#u,#head,#tool,#search,#foot{font-size:12px}.logo{width:117px;height:38px;cursor:pointer}.p1{line-height:120%;margin-left:-12pt}.p2{width:100%;line-height:120%;margin-left:-12pt}#wrapper{_zoom:1}#container{word-break:break-all;word-wrap:break-word;position:relative}.container_s{width:1002px}.container_l{width:1222px}#content_left{width:636px;float:left;padding-left:35px}#content_right{border-left:1px solid #e1e1e1;float:right}.container_s #content_right{width:271px}.container_l #content_right{width:434px}.content_none{padding-left:35px}.pc-fresh-smooth *{-webkit-font-smoothing:antialiased;-moz-osx-font-smoothing:grayscale}#u{color:#999;white-space:nowrap;position:absolute;right:10px;top:4px;z-index:299}#u a{color:#00c;margin:0 5px}#u .reg{margin:0}#u .last{margin-right:0}#u .un{font-weight:700;margin-right:5px}#u ul{width:100%;background:#fff;border:1px solid #9b9b9b}#u li{height:25px}#u li a{width:100%;height:25px;line-height:25px;display:block;text-align:left;text-decoration:none;text-indent:6px;margin:0;filter:none\9}#u li a:hover{background:#ebebeb}#u li.nl{border-top:1px solid #ebebeb}#user{display:inline-block}#user_center{position:relative;display:inline-block}#user_center .user_center_btn{margin-right:5px}.userMenu{width:64px;position:absolute;right:7px;_right:2px;top:15px;top:14px\9;*top:15px;padding-top:4px;display:none;*background:#fff}#head{padding-left:35px;margin-bottom:20px;width:900px}.fm{clear:both;position:relative;z-index:297}.nv a,.nv b,.btn,#more{font-size:14px}.s_nav{height:45px}.s_nav .s_logo{margin-right:20px;float:left}.s_nav .s_logo img{border:0;display:block}.s_tab{line-height:18px;padding:20px 0 0;float:left}.s_nav a{color:#00c;font-size:14px}.s_nav b{font-size:14px}.s_ipt_wr{width:536px;height:30px;display:inline-block;margin-right:5px;background-position:0 -96px;border:1px solid #b6b6b6;border-color:#7b7b7b #b6b6b6 #b6b6b6 #7b7b7b;vertical-align:top}.s_ipt{width:523px;height:22px;font:16px/18px arial;line-height:22px;margin:5px 0 0 7px;padding:0;background:#fff;border:0;outline:0;-webkit-appearance:none}.s_btn{width:95px;height:32px;padding-top:2px\9;font-size:14px;padding:0;background-color:#ddd;background-position:0 -48px;border:0;cursor:pointer}.s_btn_h{background-position:-240px -48px}.s_btn_wr{width:97px;height:34px;display:inline-block;background-position:-120px -48px;*position:relative;z-index:0;vertical-align:top}.yy_fm .s_ipt_wr,.yy_fm .s_ipt_wr.iptfocus,.yy_fm .s_ipt_wr:hover,.yy_fm .s_ipt_wr.ipthover{border-color:#e10602 transparent #e10602 #e10602;animation:yy-ipt .2s;-moz-animation:yy-ipt .2s;-webkit-animation:yy-ipt .2s;-o-animation:yy-ipt .2s}.yy_fm .s_btn{background-color:#e10602;border-bottom:1px solid #c30602;animation:yunying .2s;-moz-animation:yunying .2s;-webkit-animation:yunying .2s;-o-animation:yunying .2s}.yy_fm_blue .s_ipt_wr,.yy_fm_blue .s_ipt_wr.iptfocus,.yy_fm_blue .s_ipt_wr:hover,.yy_fm_blue .s_ipt_wr.ipthover{animation:yy-ipt-blue .2s;border-color:#4791ff transparent #4791ff #4791ff}.yy_fm_blue .s_btn{animation:yunying-blue .2s;background-color:#3385ff;border-bottom:1px solid #2d78f4}@keyframes yy-ipt{0%{border-color:#4791ff transparent #4791ff #4791ff}100%{border-color:#e10602 transparent #e10602 #e10602}}@-moz-keyframes yy-ipt{0%{border-color:#4791ff transparent #4791ff #4791ff}100%{border-color:#e10602 transparent #e10602 #e10602}}@-webkit-keyframes yy-ipt{0%{border-color:#4791ff transparent #4791ff #4791ff}100%{border-color:#e10602 transparent #e10602 #e10602}}@-o-keyframes yy-ipt{0%{border-color:#4791ff transparent #4791ff #4791ff}100%{border-color:#e10602 transparent #e10602 #e10602}}@keyframes yy-ipt-blue{0%{border-color:#e10602 transparent #e10602 #e10602}100%{border-color:#4791ff transparent #4791ff #4791ff}}@-moz-keyframes yy-ipt-blue{0%{border-color:#e10602 transparent #e10602 #e10602}100%{border-color:#4791ff transparent #4791ff #4791ff}}@-webkit-keyframes yy-ipt-blue{0%{border-color:#e10602 transparent #e10602 #e10602}100%{border-color:#4791ff transparent #4791ff #4791ff}}@-o-keyframes yy-ipt-blue{0%{border-color:#e10602 transparent #e10602 #e10602}100%{border-color:#4791ff transparent #4791ff #4791ff}}@keyframes yunying{0%{background-color:#3385ff;border-bottom:1px solid #2d78f4}100%{background-color:#e10602;border-bottom:1px solid #c30602}}@-moz-keyframes yunying{0%{background-color:#3385ff;border-bottom:1px solid #2d78f4}100%{background-color:#e10602;border-bottom:1px solid #c30602}}@-webkit-keyframes yunying{0%{background-color:#3385ff;border-bottom:1px solid #2d78f4}100%{background-color:#e10602;border-bottom:1px solid #c30602}}@-o-keyframes yunying{0%{background-color:#3385ff;border-bottom:1px solid #2d78f4}100%{background-color:#e10602;border-bottom:1px solid #c30602}}@keyframes yunying-blue{0%{background-color:#e10602;border-bottom:1px solid #c30602}100%{background-color:#3385ff;border-bottom:1px solid #2d78f4}}@-moz-keyframes yunying-blue{0%{background-color:#e10602;border-bottom:1px solid #c30602}100%{background-color:#3385ff;border-bottom:1px solid #2d78f4}}@-webkit-keyframes yunying-blue{0%{background-color:#e10602;border-bottom:1px solid #c30602}100%{background-color:#3385ff;border-bottom:1px solid #2d78f4}}@-o-keyframes yunying-blue{0%{background-color:#e10602;border-bottom:1px solid #c30602}100%{background-color:#3385ff;border-bottom:1px solid #2d78f4}}.sethf{padding:0;margin:0;font-size:14px}.set_h{display:none;behavior:url(#default#homepage)}.set_f{display:none}.shouji{margin-left:19px}.shouji a{text-decoration:none}#head .bdsug{top:33px}#search form{position:relative}#search form .bdsug{bottom:33px}.bdsug{display:none;position:absolute;z-index:1;width:538px;background:#fff;border:1px solid #ccc;_overflow:hidden;box-shadow:1px 1px 3px #ededed;-webkit-box-shadow:1px 1px 3px #ededed;-moz-box-shadow:1px 1px 3px #ededed;-o-box-shadow:1px 1px 3px #ededed}.bdsug.bdsugbg ul{background:url(https://pss.bdstatic.com/r/www/cache/static/protocol/https/home/img/sugbg_1762fe7.png) 100% 100% no-repeat;background-size:100px 110px;background-image:url(https://pss.bdstatic.com/r/www/cache/static/protocol/https/home/img/sugbg_90fc9cf.gif)\9}.bdsug li{width:522px;color:#000;font:14px arial;line-height:22px;padding:0 8px;position:relative;cursor:default}.bdsug li.bdsug-s{background:#f0f0f0}.bdsug-store span,.bdsug-store b{color:#7A77C8}.bdsug-store-del{font-size:12px;color:#666;text-decoration:underline;position:absolute;right:8px;top:0;cursor:pointer;display:none}.bdsug-s .bdsug-store-del{display:inline-block}.bdsug-ala{display:inline-block;border-bottom:1px solid #e6e6e6}.bdsug-ala h3{line-height:14px;background:url(//www.baidu.com/img/sug_bd.png) no-repeat left center;margin:8px 0 5px;font-size:12px;font-weight:400;color:#7B7B7B;padding-left:20px}.bdsug-ala p{font-size:14px;font-weight:700;padding-left:20px}.bdsug .bdsug-direct{width:auto;padding:0;border-bottom:1px solid #f1f1f1}.bdsug .bdsug-direct p{color:#00c;font-weight:700;line-height:34px;padding:0 8px;cursor:pointer;white-space:nowrap;overflow:hidden}.bdsug .bdsug-direct p img{width:16px;height:16px;margin:7px 6px 9px 0;vertical-align:middle}.bdsug .bdsug-direct p span{margin-left:8px}.bdsug .bdsug-direct p i{font-size:12px;line-height:100%;font-style:normal;font-weight:400;color:#fff;background-color:#2b99ff;display:inline;text-align:center;padding:1px 5px;*padding:2px 5px 0;margin-left:8px;overflow:hidden}.bdsug .bdsug-pcDirect{color:#000;font-size:14px;line-height:30px;height:30px;background-color:#f8f8f8}.bdsug .bdsug-pc-direct-tip{position:absolute;right:15px;top:8px;width:55px;height:15px;display:block;background:url(https://pss.bdstatic.com/r/www/cache/static/protocol/https/global/img/pc_direct_42d6311.png) no-repeat 0 0}.bdsug li.bdsug-pcDirect-s{background-color:#f0f0f0}.bdsug .bdsug-pcDirect-is{color:#000;font-size:14px;line-height:22px;background-color:#f8f8f8}.bdsug .bdsug-pc-direct-tip-is{position:absolute;right:15px;top:3px;width:55px;height:15px;display:block;background:url(https://pss.bdstatic.com/r/www/cache/static/protocol/https/global/img/pc_direct_42d6311.png) no-repeat 0 0}.bdsug li.bdsug-pcDirect-is-s{background-color:#f0f0f0}.bdsug .bdsug-pcDirect-s .bdsug-pc-direct-tip,.bdsug .bdsug-pcDirect-is-s .bdsug-pc-direct-tip-is{background-position:0 -15px}.bdsug .bdsug-newicon{color:#929292;opacity:.7;font-size:12px;display:inline-block;line-height:22px;letter-spacing:2px}.bdsug .bdsug-s .bdsug-newicon{opacity:1}.bdsug .bdsug-newicon i{letter-spacing:0;font-style:normal}.bdsug .bdsug-feedback-wrap{text-align:right;background:#fafafa;color:#666;height:25px;line-height:27px}.bdsug .bdsug-feedback{margin-right:10px;text-decoration:underline;color:#666}.toggle-underline{text-decoration:none}.toggle-underline:hover{text-decoration:underline}#tb_mr{color:#00c;cursor:pointer;position:relative;z-index:298}#tb_mr b{font-weight:400;text-decoration:underline}#tb_mr small{font-size:11px}#rs{width:900px;background:#fff;padding:8px 0;margin:20px 0 0 15px}#rs td{width:5%}#rs th{font-size:14px;font-weight:400;line-height:19px;white-space:nowrap;text-align:left;vertical-align:top}#rs .tt{font-weight:700;padding:0 10px 0 20px}#rs .tt_normal{font-weight:400}#rs_top{font-size:14px;margin-bottom:22px}#rs_top a{margin-right:18px}#container .rs{margin:30px 0 20px;padding:5px 0 15px;font-size:14px;width:540px;padding-left:121px;position:relative;background-color:#fafafa}#container .noback{background-color:#fff}#content_left .rs{margin-left:-121px}#container .rs table{width:540px}#container .rs td{width:5px}#container .rs th{font-size:14px;font-weight:400;white-space:nowrap;text-align:left;vertical-align:top;width:175px;line-height:22px}#container .rs .tt{font-weight:700;padding:0 10px 0 20px;padding:0;line-height:30px;font-size:16px}#container .rs a{margin:0;height:24px;width:173px;display:inline-block;line-height:25px;border:1px solid #ebebeb;text-align:center;vertical-align:middle;overflow:hidden;outline:0;color:#333;background-color:#fff;text-decoration:none}#container .rs a:hover{border-color:#388bff}.c-tip-con .c-tip-menu-b ul{width:100px}.c-tip-con .c-tip-menu-b ul{text-align:center}.c-tip-con .c-tip-menu-b li a{display:block;text-decoration:none;cursor:pointer;background-color:#fff;padding:3px 0;color:#666}.c-tip-con .c-tip-menu-b li a:hover{display:block;background-color:#ebebeb}.c-tip-con.baozhang-r-tip{visibility:hidden}.aviation-new a{background:0 0;color:#91B9F7;font-size:16px;width:16px;height:auto;vertical-align:top}.c-tip-con.custom-wrap-tip .c-tip-info{width:auto}.c-tip-con.aviation-wrap-tip{box-shadow:0 2px 10px 0 rgba(0,0,0,.1);border-radius:12px;border:0;padding:12px}.c-tip-con.aviation-wrap-tip .c-tip-info{margin:0;width:auto}.c-tip-con.aviation-wrap-tip .c-tip-item-i{padding:0;line-height:1}.c-tip-con.aviation-wrap-tip .c-tip-item-i .c-tip-item-icon{margin-left:0}.c-tip-con.aviation-wrap-tip .aviation-title{line-height:1}#search{width:900px;padding:35px 0 16px 35px}#search .s_help{position:relative;top:10px}.site_tip{font-size:12px;margin-bottom:20px}.site_tip_icon{width:56px;height:56px;background:url(//www.baidu.com/aladdin/img/tools/tools-3.png) -288px 0 no-repeat}.to_zhidao,.to_tieba,.to_zhidao_bottom{font-size:16px;line-height:24px;margin:20px 0 0 35px}.to_tieba .c-icon-tieba{float:left}.f{line-height:115%;*line-height:120%;font-size:100%;width:33.7em;word-break:break-all;word-wrap:break-word}.h{margin-left:8px;width:100%}.r{word-break:break-all;cursor:hand;width:238px}.t{font-weight:400;font-size:medium;margin-bottom:1px}.pl{padding-left:3px;height:8px;padding-right:2px;font-size:14px}.mo,a.mo:link,a.mo:visited{color:#666;font-size:100%;line-height:10px}.htb{margin-bottom:5px}.jc a{color:#c00}a font[size=&quot;3&quot;] font,font[size=&quot;3&quot;] a font{text-decoration:underline}div.blog,div.bbs{color:#707070;padding-top:2px;font-size:13px}.result{width:33.7em;table-layout:fixed}.result-op .f{word-wrap:normal}.nums{font-size:12px;color:#999}.tools{position:absolute;top:10px;white-space:nowrap}#mHolder{width:62px;position:relative;z-index:296;top:-18px;margin-left:9px;margin-right:-12px;display:none}#mCon{height:18px;position:absolute;top:3px;top:6px\9;cursor:pointer;line-height:18px}.wrapper_l #mCon{right:7px}#mCon span{color:#00c;display:block}#mCon .hw{text-decoration:underline;cursor:pointer;display:inline-block}#mCon .pinyin{display:inline-block}#mCon .c-icon-chevron-unfold2{margin-left:5px}#mMenu{width:56px;border:1px solid #9b9b9b;position:absolute;right:7px;top:23px;display:none;background:#fff}#mMenu a{width:100%;height:100%;color:#00c;display:block;line-height:22px;text-indent:6px;text-decoration:none;filter:none\9}#mMenu a:hover{background:#ebebeb}#mMenu .ln{height:1px;background:#ebebeb;overflow:hidden;font-size:1px;line-height:1px;margin-top:-1px}.op_LAMP{background:url(//www.baidu.com/cache/global/img/aladdinIcon-1.0.gif) no-repeat 0 2px;color:#77C;display:inline-block;font-size:13px;height:12px;*height:14px;width:16px;text-decoration:none;zoom:1}.EC_mr15{margin-left:0}.pd15{padding-left:0}.map_1{width:30em;font-size:80%;line-height:145%}.map_2{width:25em;font-size:80%;line-height:145%}.favurl{background-repeat:no-repeat;background-position:0 1px;padding-left:20px}.dan_tip{font-size:12px;margin-top:4px}.dan_tip a{color:#b95b07}#more,#u ul,#mMenu,.msg_holder{box-shadow:1px 1px 2px #ccc;-moz-box-shadow:1px 1px 2px #ccc;-webkit-box-shadow:1px 1px 2px #ccc;filter:progid:DXImageTransform.Microsoft.Shadow(Strength=2, Direction=135, Color=#cccccc)\9}.hit_top{line-height:18px;margin:0 15px 10px 0;width:516px}.hit_top .c-icon-bear{height:18px;margin-right:4px}#rs_top_new,.hit_top_new{width:538px;font-size:13px;line-height:1.54;word-wrap:break-word;word-break:break-all;margin:0 0 14px}.zhannei-si{margin:0 0 10px 121px}.zhannei-si-none{margin:10px 0 -10px 121px}.zhannei-search{margin:10px 0 0 121px;color:#999;font-size:14px}.f a font[size=&quot;3&quot;] font,.f font[size=&quot;-1&quot;] a font{text-decoration:underline}h3 a font{text-decoration:underline}.c-title{font-weight:400;font-size:16px}.c-title-size{font-size:16px}.c-abstract{font-size:13px}.c-abstract-size{font-size:13px}.c-showurl{color:green;font-size:13px}.c-showurl-color{color:green}.c-cache-color{color:#666}.c-lightblue{color:#77c}.c-highlight-color{color:#c00}.c-clearfix:after{content:&quot;.&quot;;display:block;height:0;clear:both;visibility:hidden}.c-clearfix{zoom:1}.c-wrap{word-break:break-all;word-wrap:break-word}.c-icons-outer{overflow:hidden;display:inline-block;vertical-align:bottom;*vertical-align:-1px;_vertical-align:bottom}.c-icons-inner{margin-left:-4px;display:inline-block}.c-container table.result,.c-container table.result-op{width:100%}.c-container td.f{font-size:13px;line-height:1.54;width:auto}.c-container .vd_newest_main{width:auto}.c-customicon{display:inline-block;width:16px;height:16px;vertical-align:text-bottom;font-style:normal;overflow:hidden}.c-tip-icon i{display:inline-block;cursor:pointer}.c-tip-con{position:absolute;z-index:1;top:22px;left:-35px;background:#fff;border:1px solid #dcdcdc;border:1px solid rgba(0,0,0,.2);-webkit-transition:opacity .218s;transition:opacity .218s;-webkit-box-shadow:0 2px 4px rgba(0,0,0,.2);box-shadow:0 2px 4px rgba(0,0,0,.2);padding:5px 0;display:none;font-size:12px;line-height:20px}.c-tip-arrow{width:0;height:0;font-size:0;line-height:0;display:block;position:absolute;top:-16px}.c-tip-arrow-down{top:auto;bottom:0}.c-tip-arrow em,.c-tip-arrow ins{width:0;height:0;font-size:0;line-height:0;display:block;position:absolute;border:8px solid transparent;border-style:dashed dashed solid}.c-tip-arrow em{border-bottom-color:#d8d8d8}.c-tip-arrow ins{border-bottom-color:#fff;top:2px}.c-tip-arrow-down em,.c-tip-arrow-down ins{border-style:solid dashed dashed;border-color:transparent}.c-tip-arrow-down em{border-top-color:#d8d8d8}.c-tip-arrow-down ins{border-top-color:#fff;top:-2px}.c-tip-arrow .c-tip-arrow-r{border-bottom-color:#82c9fa;top:2px}.c-tip-arrow-down .c-tip-arrow-r{border-bottom-color:transparent;top:-2px}.c-tip-arrow .c-tip-arrow-c{border-bottom-color:#fecc47;top:2px}.c-tip-arrow-down .c-tip-arrow-c{border-bottom-color:transparent;top:-2px}.c-tip-con h3{font-size:12px}.c-tip-con .c-tip-title{margin:0 10px;display:inline-block;width:239px}.c-tip-con .c-tip-info{color:#666;margin:0 10px 1px;width:239px}.c-tip-con .c-tip-cer{width:370px;color:#666;margin:0 10px 1px}.c-tip-con .c-tip-title{width:auto;_width:354px}.c-tip-con .c-tip-item-i{padding:3px 0 3px 20px;line-height:14px}.c-tip-con .c-tip-item-i .c-tip-item-icon{margin-left:-20px}.c-tip-con .c-tip-menu ul{width:74px}.c-tip-con .c-tip-menu ul{text-align:center}.c-tip-con .c-tip-menu li a{display:block;text-decoration:none;cursor:pointer;background-color:#fff;padding:3px 0;color:#0000d0}.c-tip-con .c-tip-menu li a:hover{display:block;background-color:#ebebeb}.c-tip-con .c-tip-notice{width:239px;padding:0 10px}.c-tip-con .c-tip-notice .c-tip-notice-succ{color:#4cbd37}.c-tip-con .c-tip-notice .c-tip-notice-fail{color:#f13F40}.c-tip-con .c-tip-notice .c-tip-item-succ{color:#444}.c-tip-con .c-tip-notice .c-tip-item-fail{color:#aaa}.c-tip-con .c-tip-notice .c-tip-item-fail a{color:#aaa}.c-tip-close{right:10px;position:absolute;cursor:pointer}.ecard{height:86px;overflow:hidden}.c-tools{display:inline}.c-tools-share{width:239px;padding:0 10px}.c-fanyi{display:none;width:20px;height:20px;border:solid 1px #d1d1d1;cursor:pointer;position:absolute;margin-left:516px;text-align:center;color:#333;line-height:22px;opacity:.9;background-color:#fff}.c-fanyi:hover{background-color:#39f;color:#fff;border-color:#39f;opacity:1}.c-fanyi-title,.c-fanyi-abstract{display:none}.icp_info{color:#666;margin-top:2px;font-size:13px}.icon-gw,.icon-unsafe-icon{background:#2c99ff;vertical-align:text-bottom;*vertical-align:baseline;height:16px;padding-top:0;padding-bottom:0;padding-left:6px;padding-right:6px;line-height:16px;_padding-top:2px;_height:14px;_line-height:14px;font-size:12px;font-family:simsun;margin-left:10px;overflow:hidden;display:inline-block;-moz-border-radius:1px;-webkit-border-radius:1px;border-radius:1px;color:#fff}a.icon-gw{color:#fff;background:#2196ff;text-decoration:none;cursor:pointer}a.icon-gw:hover{background:#1e87ef}a.icon-gw:active{height:15px;_height:13px;line-height:15px;_line-height:13px;padding-left:5px;background:#1c80d9;border-left:1px solid #145997;border-top:1px solid #145997}.icon-unsafe-icon{background:#e54d4b}#con-at{padding-left:121px}#con-at .result-op{font-size:13px;line-height:1.52em}.wrapper_l #con-at .result-op{width:1058px}.wrapper_s #con-at .result-op{width:869px}#con-ar{margin-bottom:40px}#con-ar .result-op{margin-bottom:28px;font-size:13px;line-height:1.52em}.result_hidden{position:absolute;top:-10000px;left:-10000px}.pc-fresh-wrapper-con #con-at,.pc-fresh-wrapper-ext #con-at{padding-left:0}.pc-fresh-wrapper-con .wrapper_l #con-at .result-op,.pc-fresh-wrapper-ext .wrapper_l #con-at .result-op,.pc-fresh-wrapper-con .wrapper_s #con-at .result-op,.pc-fresh-wrapper-ext .wrapper_s #con-at .result-op{width:100vw}@media screen and (min-width:1380px){html{overflow-y:auto;overflow-x:hidden}body.pc-fresh-wrapper-con{width:100vw;overflow:hidden;min-height:500px}}@media screen and (min-width:1412px){body.pc-fresh-wrapper-ext{width:100vw;overflow:hidden;min-height:500px}}#wrapper_wrapper_box{padding-top:9px}#content_left .result-op,#content_left .result{margin-bottom:14px;border-collapse:collapse}#content_left .c-border .result-op,#content_left .c-border .result{margin-bottom:25px}#content_left .c-border .result-op:last-child,#content_left .c-border .result:last-child{margin-bottom:12px}#content_left .result .f,#content_left .result-op .f{padding:0}.subLink_factory{border-collapse:collapse}.subLink_factory td{padding:0}.subLink_factory td.middle,.subLink_factory td.last{color:#666}.subLink_factory td a{text-decoration:underline}.subLink_factory td.rightTd{text-align:right}.subLink_factory_right{width:100%}.subLink_factory_left td{padding-right:26px}.subLink_factory_left td.last{padding:0}.subLink_factory_left td.first{padding-right:75px}.subLink_factory_right td{width:90px}.subLink_factory_right td.first{width:auto}.subLink_answer{padding-top:4px}.subLink_answer li{margin-bottom:4px}.subLink_answer h4{margin:0;padding:0;font-weight:400}.subLink_answer .label_wrap span{display:inline-block;color:#9195A3;margin-right:8px}.subLink_answer .label_wrap span em{color:#666;padding-left:8px}.subLink_answer span.c-icon{margin-right:4px}.subLink_answer_dis{padding:0 3px}.subLink_answer .date{color:#666}.general_image_pic a{background:#fff no-repeat center center;text-decoration:none;display:block;overflow:hidden;text-align:left}.res_top_banner{height:36px;text-align:left;border-bottom:1px solid #e3e3e3;background:#f7f7f7;font-size:13px;padding-left:8px;color:#333;position:relative;z-index:302}.res_top_banner span{_zoom:1}.res_top_banner .res_top_banner_icon{background-position:0 -216px;width:18px;height:18px;margin:9px 10px 0 0}.res_top_banner .res_top_banner_icon_baiduapp{background:url(https://pss.bdstatic.com/r/www/cache/static/protocol/https/global/img/baiduappLogo_de45621.png) no-repeat 0 0;width:24px;height:24px;margin:3px 10px 0 0;position:relative;top:3px}.res_top_banner .res_top_banner_icon_windows{background:url(https://pss.bdstatic.com/r/www/cache/static/protocol/https/global/img/winlogo_e925689.png) no-repeat 0 0;width:18px;height:18px;margin:9px 10px 0 0}.res_top_banner .res_top_banner_download{display:inline-block;width:65px;line-height:21px;_padding-top:1px;margin:0 0 0 10px;color:#333;background:#fbfbfb;border:1px solid #b4b6b8;text-align:center;text-decoration:none}.res_top_banner .res_top_banner_download:hover{border:1px solid #38f}.res_top_banner .res_top_banner_download:active{background:#f0f0f0;border:1px solid #b4b6b8}.res_top_banner .res_top_banner_close{background-position:-672px -144px;cursor:pointer;position:absolute;right:10px;top:10px}.res_top_banner_for_win{height:34px;text-align:left;border-bottom:1px solid #f0f0f0;background:#fdfdfd;font-size:13px;padding-left:12px;color:#333;position:relative;z-index:302}.res_top_banner_for_win span{_zoom:1;color:#666}.res_top_banner_for_win .res_top_banner_download{display:inline-block;width:auto;line-height:21px;_padding-top:1px;margin:0 0 0 16px;color:#333;text-align:left;text-decoration:underline}.res_top_banner_for_win .res_top_banner_icon_windows{background:url(https://pss.bdstatic.com/r/www/cache/static/protocol/https/global/img/winlogo_e925689.png) no-repeat 0 0;width:18px;height:18px;margin:8px 8px 0 0}.res_top_banner_for_win .res_top_banner_close{background-position:-672px -144px;cursor:pointer;position:absolute;right:10px;top:10px}.res-gap-right16{margin-right:16px}.res-border-top{border-top:1px solid #f3f3f3}.res-border-bottom{border-bottom:1px solid #f3f3f3}.res-queryext-pos{position:relative;top:1px;_top:0}.res-queryext-pos-new{position:relative;top:-2px;_top:0}.c-trust-ecard{height:86px;_height:97px;overflow:hidden}.op-recommend-sp-gap{margin-right:6px}@-moz-document url-prefix(){.result,.f{width:538px}}#ftCon{display:none}#qrcode{display:none}#pad-version{display:none}#index_guide{display:none}#index_logo{display:none}#u1{display:none}.s_ipt_wr{height:32px}body{padding:0}.s_form:after,.s_tab:after{content:&quot;.&quot;;display:block;height:0;clear:both;visibility:hidden}.s_form{zoom:1;height:55px;padding:0 0 0 10px}#result_logo{float:left;margin:7px 0 0}#result_logo img{width:101px;height:33px}#result_logo.qm-activity{filter:progid:DXImageTransform.Microsoft.BasicImage(grayscale=1);-webkit-filter:grayscale(100%);-moz-filter:grayscale(100%);-ms-filter:grayscale(100%);-o-filter:grayscale(100%);filter:grayscale(100%);filter:gray}#head{padding:0;margin:0;width:100%;position:absolute;z-index:301;min-width:1000px;background:#fff;border-bottom:1px solid #ebebeb;position:fixed;_position:absolute;-webkit-transform:translateZ(0)}#head .head_wrapper{_width:1000px}#head.s_down{box-shadow:0 0 5px #888}.fm{clear:none;float:left;margin:11px 0 0 10px}#s_tab{background:#f8f8f8;line-height:36px;height:38px;padding:55px 0 0 121px;float:none;zoom:1}#s_tab .s-old-tag a,#s_tab .s-old-tag b{width:54px;display:inline-block;text-decoration:none;text-align:center;color:#666;font-size:14px}#s_tab .s-old-tag b{border-bottom:2px solid #38f;font-weight:700;color:#323232}#s_tab .s-old-tag a:hover{color:#323232}#content_left{width:540px;padding-left:121px;padding-top:5px}.new_input_wrapper #content_left{padding-top:8px}#content_right{margin-top:45px}.sam_newgrid #content_right{margin-top:40px}.pc-fresh-wrapper .sam_newgrid #content_right{margin-top:21px}#content_bottom{width:540px;padding-left:121px}.to_tieba,.to_zhidao_bottom{margin:10px 0 0 121px;padding-top:5px}.nums{margin:0 0 0 121px;height:42px;line-height:42px}.new_nums{font-size:13px;height:41px;line-height:41px}#rs{padding:0;margin:6px 0 0 121px;width:600px}#rs th{width:175px;line-height:22px}#rs .tt{padding:0;line-height:30px}#rs td{width:5px}#rs table{width:540px}#help a.emphasize{font-weight:700;text-decoration:underline}.content_none{padding:45px 0 25px 121px;float:left;width:560px}.nors p{font-size:18px;color:#000}.nors p em{color:#c00}.nors .tip_head{color:#666;font-size:13px;line-height:28px}.nors li{color:#333;line-height:28px;font-size:13px;list-style-type:none}#mCon{top:5px}.s_ipt_wr.bg,.s_btn_wr.bg,#su.bg{background-image:none}.s_btn_wr{width:auto;height:auto;border-bottom:1px solid transparent;*border-bottom:0}.s_btn{width:100px;height:34px;color:#fff;letter-spacing:1px;background:#3385ff;border-bottom:1px solid #2d78f4;outline:medium;*border-bottom:0;-webkit-appearance:none;-webkit-border-radius:0}.s_btn.btnhover{background:#317ef3;border-bottom:1px solid #2868c8;*border-bottom:0;box-shadow:1px 1px 1px #ccc}.s_btn_h{background:#3075dc;box-shadow:inset 1px 1px 3px #2964bb;-webkit-box-shadow:inset 1px 1px 3px #2964bb;-moz-box-shadow:inset 1px 1px 3px #2964bb;-o-box-shadow:inset 1px 1px 3px #2964bb}.yy_fm .s_btn.btnhover{background:#D10400;border-bottom:1px solid #D10400}.yy_fm .s_btn_h{background:#C00400;box-shadow:inset 1px 1px 3px #A00300;-webkit-box-shadow:inset 1px 1px 3px #A00300}#wrapper_wrapper .container_l .EC_ppim_top,#wrapper_wrapper .container_xl .EC_ppim_top{width:640px}#wrapper_wrapper .container_s .EC_ppim_top{width:570px}#head .c-icon-bear-round{display:none}.container_l #content_right{width:384px}.container_l{width:1212px}.container_xl #content_right{width:384px}.container_xl{width:1257px}.index_tab_top{display:none}.index_tab_bottom{display:none}#lg{display:none}#m{display:none}#ftCon{display:none}#ent_sug{position:absolute;margin:141px 0 0 130px;font-size:13px;color:#666}.pc-fresh-wrapper #ent_sug{width:var(--wrapper-width);left:50%;transform:translateX(-50%);margin:141px 0 0}.foot_fixed_bottom{position:fixed;bottom:0;width:100%;_position:absolute;_bottom:auto}#head .headBlock{margin:-5px 0 6px 121px}#content_left .leftBlock{margin-bottom:14px;padding-bottom:5px;border-bottom:1px solid #f3f3f3}.hint_toprq_tips{position:relative;width:537px;height:19px;line-height:19px;overflow:hidden;display:none}.hint_toprq_tips span{color:#666}.hint_toprq_icon{margin:0 4px 0 0}.hint_toprq_tips_items{width:444px;_width:440px;max-height:38px;position:absolute;left:95px;top:1px}.hint_toprq_tips_items div{display:inline-block;float:left;height:19px;margin-right:18px;white-space:nowrap;word-break:keep-all}.translateContent{max-width:350px}.translateContent .translateTool{height:16px;margin:-3px 2px}.translateContent .action-translate,.translateContent .action-search{display:inline-block;width:20px;height:16px;background:url(https://pss.bdstatic.com/r/www/cache/static/protocol/https/global/img/translate_tool_icon_57087b6.gif) no-repeat}.translateContent .action-translate{background-position:0 0;border-right:1px solid #dcdcdc}.translateContent .action-translate:hover{background-position:0 -20px}.translateContent .action-search{background-position:-20px 0}.translateContent .action-search:hover{background-position:-20px -20px}.nums{width:538px}.search_tool{_padding-top:15px}.head_nums_cont_outer{height:40px;overflow:hidden;position:relative}.new_head_nums_cont_outer{height:35px}.head_nums_cont_inner{position:relative}.search_tool_conter .c-gap-left{margin-left:23px}.search_tool_conter .c-icon-triangle-down{opacity:.6}.search_tool_conter .c-icon-triangle-down:hover{opacity:1}.search_tool,.search_tool_close{float:right}.search_tool,.search_tool_conter span{cursor:pointer;color:#666}.search_tool:hover,.search_tool_conter span:hover{color:#333}.search_tool_conter{font-size:12px;color:#666;margin:0 0 0 121px;height:42px;width:538px;line-height:42px;*height:auto;*line-height:normal;*padding:14px 0}.new_search_tool_conter{font-size:12px;color:#666;margin:0 0 0 121px;height:41px;width:538px;line-height:39px;*height:auto;*line-height:normal;*padding:14px 0}.search_tool_conter span strong{color:#666}.c-tip-con .c-tip-langfilter ul{width:80px;text-align:left;color:#666}.c-tip-con .c-tip-langfilter li a{text-indent:15px;color:#666}.c-tip-con .c-tip-langfilter li span{text-indent:15px;padding:3px 0;color:#999;display:block}.c-tip-con .c-tip-timerfilter ul{width:117px;text-align:left;color:#666}.c-tip-con .c-tip-timerfilter-ft ul{width:180px}.c-tip-con .c-tip-timerfilter-si ul{width:206px;padding:7px 10px 10px}.c-tip-con .c-tip-timerfilter li a{text-indent:15px;color:#666}.c-tip-con .c-tip-timerfilter li span{text-indent:15px;padding:3px 0;color:#999;display:block}.c-tip-con .c-tip-timerfilter-ft li a,.c-tip-con .c-tip-timerfilter-ft li span{text-indent:20px}.c-tip-custom{padding:0 15px 10px;position:relative;zoom:1}.c-tip-custom hr{border:0;height:0;border-top:1px solid #ebebeb}.c-tip-custom p{color:#b6b6b6;height:25px;line-height:25px;margin:2px 0}.c-tip-custom .c-tip-custom-et{margin-bottom:7px}.c-tip-custom-input,.c-tip-si-input{display:inline-block;font-size:11px;color:#333;margin-left:4px;padding:0 2px;width:74%;height:16px;line-height:16px\9;border:1px solid #ebebeb;outline:0;box-sizing:content-box;-webkit-box-sizing:content-box;-moz-box-sizing:content-box;overflow:hidden;position:relative}.c-tip-custom-input-init{color:#d4d4d4}.c-tip-custom-input-focus,.c-tip-si-input-focus{border:1px solid #3385ff}.c-tip-timerfilter-si .c-tip-si-input{width:138px;height:22px;line-height:22px;vertical-align:0;*vertical-align:-6px;_vertical-align:-5px;padding:0 5px;margin-left:0}.c-tip-con .c-tip-timerfilter li .c-tip-custom-submit,.c-tip-con .c-tip-timerfilter li .c-tip-timerfilter-si-submit{display:inline;padding:4px 10px;margin:0;color:#333;border:1px solid #d8d8d8;font-family:inherit;font-weight:400;text-align:center;vertical-align:0;background-color:#f9f9f9;outline:0}.c-tip-con .c-tip-timerfilter li .c-tip-custom-submit:hover,.c-tip-con .c-tip-timerfilter li .c-tip-timerfilter-si-submit:hover{display:inline;border-color:#388bff}.c-tip-timerfilter-si-error,.c-tip-timerfilter-custom-error{display:none;color:#3385FF;padding-left:4px}.c-tip-timerfilter-custom-error{padding:0;margin:-5px -13px 7px 0}#c-tip-custom-calenderCont{position:absolute;background:#fff;white-space:nowrap;padding:5px 10px;color:#000;border:1px solid #e4e4e4;-webkit-box-shadow:0 2px 4px rgba(0,0,0,.2);box-shadow:0 2px 4px rgba(0,0,0,.2)}#c-tip-custom-calenderCont p{text-align:center;padding:2px 0 4px;*padding:4px 0}#c-tip-custom-calenderCont p i{color:#8e9977;cursor:pointer;text-decoration:underline;font-size:13px}#c-tip-custom-calenderCont .op_cal{background:#fff}.op_cal table{background:#eeefea;margin:0;border-collapse:separate}.op_btn_pre_month,.op_btn_next_month{cursor:pointer;display:block;margin-top:6px}.op_btn_pre_month{float:left;background-position:0 -46px}.op_btn_next_month{float:right;background-position:-18px -46px}.op_cal .op_mon_pre1{padding:0}.op_mon th{text-align:center;font-size:12px;background:#FFF;font-weight:700;border:1px solid #FFF;padding:0}.op_mon td{text-align:center;cursor:pointer}.op_mon h5{margin:0;padding:0 4px;text-align:center;font-size:14px;background:#FFF;height:28px;line-height:28px;border-bottom:1px solid #f5f5f5;margin-bottom:5px}.op_mon strong{font-weight:700}.op_mon td{padding:0 5px;border:1px solid #fff;font-size:12px;background:#fff;height:100%}.op_mon td.op_mon_pre_month{color:#a4a4a4}.op_mon td.op_mon_cur_month{color:#00c}.op_mon td.op_mon_next_month{color:#a4a4a4}.op_mon td.op_mon_day_hover{color:#000;border:1px solid #278df2}.op_mon td.op_mon_day_selected{color:#FFF;border:1px solid #278df2;background:#278df2}.op_mon td.op_mon_day_disabled{cursor:not-allowed;color:#ddd}.zhannei-si-none,.zhannei-si,.hit_quet,.zhannei-search{display:none}#c-tip-custom-calenderCont .op_mon td.op_mon_cur_month{color:#000}#c-tip-custom-calenderCont .op_mon td.op_mon_day_selected{color:#fff}.c-icon-toen{width:24px;height:24px;line-height:24px;background-color:#1cb7fd;color:#fff;font-size:14px;font-weight:700;font-style:normal;display:block;display:inline-block;float:left;text-align:center}.hint_common_restop{width:538px;color:#999;font-size:12px;text-align:left;margin:5px 0 10px 121px}.hint_common_restop.hint-adrisk-pro{margin-top:4px;margin-bottom:13px}.hint_common_restop .hint-adrisk-title{color:#333;margin-bottom:3px}#con-at~#wrapper_wrapper .hint_common_restop{padding-top:7px}.sitelink{overflow:auto;zoom:1}.sitelink_summary{float:left;width:47%;padding-right:30px}.sitelink_summary a{font-size:1.1em;position:relative}.sitelink_summary_last{padding-right:0}.sitelink_en{overflow:auto;zoom:1}.sitelink_en_summary{float:left;width:47%;padding-right:30px}.sitelink_en_summary a{font-size:1.1em;position:relative}.sitelink_en_summary_last{padding-right:0}.sitelink_en_summary_title,.sitelink_en_summary .m{height:22px;overflow:hidden}.without-summary-sitelink-en-container{overflow:hidden;height:22px}.without-summary-sitelink-en{float:left}.without-summary-sitelink-en-delimiter{margin-right:5px;margin-left:5px}.wise-qrcode-wrapper{height:42px;line-height:42px;position:absolute;margin-left:8px;top:0;z-index:300}.wise-qrcode-icon-outer{overflow:hidden}.wise-qrcode-icon{position:relative;display:inline-block;width:15px;height:15px;vertical-align:text-bottom;overflow:hidden;opacity:.5;background:url(https://pss.bdstatic.com/r/www/cache/static/protocol/https/global/img/qrcode_icon_ae03227.png) no-repeat;-webkit-transform:translateY(42px);-ms-transform:translateY(42px);transform:translateY(42px);-webkit-background-size:100% 100%;background-size:100%}.wise-qrcode-container{padding:15px;background:#fff;display:none;top:61px;left:0;-webkit-transform:translateX(-50%);-ms-transform:translateX(-50%);transform:translateX(-50%);-webkit-box-shadow:0 0 1px rgba(0,0,0,.5);box-shadow:0 0 1px rgba(0,0,0,.5)}.wise-qrcode-wrapper.show:hover .wise-qrcode-container{display:block}.wise-qrcode-image{width:90px;height:90px;display:inline-block;vertical-align:middle}.wise-qrcode-image .wise-qrcode-canvas{width:100%;height:100%}.wise-qrcode-right{display:inline-block;vertical-align:middle;margin-left:15px}.wise-qrcode-title{font-size:16px;color:#000;line-height:26px}.wise-qrcode-text{font-size:12px;line-height:22px;color:#555}.pc-fresh-card #container.sam_newgrid #content_left .result-op,.pc-fresh-card #container.sam_newgrid #content_left .result{margin-bottom:27px}.pc-fresh-wrapper-ext,.pc-fresh-wrapper-con{--wrapper-padding:152px;--container-left-gap:152px}.pc-fresh-wrapper-ext #container.sam_newgrid,.pc-fresh-wrapper-con #container.sam_newgrid{margin:0 auto;padding:0 var(--wrapper-padding)}.pc-fresh-wrapper-ext #content_left,.pc-fresh-wrapper-con #content_left{--content-padding-top:18px;padding-left:0!important;margin-left:0!important;padding-top:var(--content-padding-top)}.pc-fresh-wrapper-ext #content_left:has(.c-group-wrapper),.pc-fresh-wrapper-con #content_left:has(.c-group-wrapper),.pc-fresh-wrapper-ext #content_left:has(.result-op:first-child .cu-border.sc-cover-card),.pc-fresh-wrapper-con #content_left:has(.result-op:first-child .cu-border.sc-cover-card),.pc-fresh-wrapper-ext #content_left:has(.result-op:first-child .cosc-card .cosc-card-shadow),.pc-fresh-wrapper-con #content_left:has(.result-op:first-child .cosc-card .cosc-card-shadow){--content-padding-top:21px}.pc-fresh-wrapper-ext{--container-gap:124px;--wrapper-width:1108px;--left-container-width:660px;--right-container-width:324px;--pc-col-gap:12px}.pc-fresh-wrapper-con{--container-gap:120px;--wrapper-width:1076px;--left-container-width:608px;--right-container-width:348px;--pc-col-gap:16px}@media screen and (min-width:1412px){.pc-fresh-wrapper-ext{--container-left-gap:calc(50vw - var(--wrapper-width) / 2)}}@media screen and (min-width:1380px){.pc-fresh-wrapper-con{--container-left-gap:calc(50vw - var(--wrapper-width) / 2)}}#container.sam_newgrid{margin-left:150px}#container.sam_newgrid #content_right{outline:0;border-left:0;padding:0}#container.sam_newgrid #content_right.topic-gap{margin-top:13px}#container.sam_newgrid .content-right-hidden{visibility:hidden}#container.sam_newgrid #content_left{outline:0;padding-left:0}#container.sam_newgrid #content_left .result-op,#container.sam_newgrid #content_left .result{margin-bottom:20px}#container.sam_newgrid #con-ar .result-op{margin-bottom:20px;line-height:21px}#container.sam_newgrid .c-container .t,#container.sam_newgrid .c-container .c-title{margin-bottom:4px}#container.sam_newgrid .c-container .t a,#container.sam_newgrid .c-container .c-title a{display:inline-block;text-decoration:underline}#container.sam_newgrid .c-container .t a em,#container.sam_newgrid .c-container .c-title a em{text-decoration:underline}#container.sam_newgrid .c-container .t.c-title-border-gap,#container.sam_newgrid .c-container .c-title.c-title-border-gap{margin-bottom:8px}#container.sam_newgrid a .t,#container.sam_newgrid a .c-title{text-decoration:underline}#container.sam_newgrid a .t em,#container.sam_newgrid a .c-title em{text-decoration:underline}#container.sam_newgrid .hint_common_restop,#container.sam_newgrid .nums,#container.sam_newgrid #rs,#container.sam_newgrid .search_tool_conter{margin-left:0}#container.sam_newgrid .content_none{padding-left:0}#container.sam_newgrid .result .c-tools,#container.sam_newgrid .result-op .c-tools{margin-left:8px;cursor:pointer}#container.sam_newgrid .result .c-tools .c-icon,#container.sam_newgrid .result-op .c-tools .c-icon{font-size:13px;color:rgba(0,0,0,.1);height:17px;width:13px;text-decoration:none;overflow:visible}#container.sam_newgrid .se_st_footer .c-tools .c-icon{position:relative;top:-1px}#container.sam_newgrid .c-showurl{color:#626675;font-family:Arial,sans-serif}#container.sam_newgrid .c-showurl-hover{text-decoration:underline;color:#315efb}#container.sam_newgrid .c-showem{text-decoration:underline;color:#f73131}#container.sam_newgrid .c-icons-inner{margin-left:0}#container.sam_newgrid .c-trust-as{cursor:pointer}#container.sam_newgrid .c-icon-xls-new{color:#8bba75}#container.sam_newgrid .c-icon-txt-new{color:#708cf6}#container.sam_newgrid .c-icon-pdf-new{color:#e56755}#container.sam_newgrid .c-icon-ppt-new{color:#e27c59}#container.sam_newgrid .c-icon-doc-new{color:#509de0}#container.sam_newgrid .se-st-default-abs-icon{font-size:16px;width:16px;height:18px}#container.sam_newgrid .se-st-default-t-icon{width:20px;height:22px;position:relative;font-size:20px;top:-1px}#container.sam_newgrid .right-fixed{position:fixed;top:86px;z-index:1}#container.sam_newgrid .right-fixed.fixed-bottom{bottom:88px;top:auto}#container.sam_newgrid .right-ceiling{position:fixed;top:98px}#container.sam_newgrid .right-ceiling-has-tag{position:fixed;top:148px}.new-pmd .subLink_answer{padding-top:3px}.new-pmd .subLink_answer li{margin-bottom:5px}.new-pmd .subLink_answer li:last-child{margin-bottom:4px}.new-pmd .normal-gf-icon{font-size:12px;padding:0 3px;position:relative;top:-3px}.new-pmd .sitelink_summary{width:272px;padding-right:16px}.new-pmd .sitelink_summary_last{padding-right:0}.new-pmd.bd_weixin_popup .c-tips-icon-close{font-size:16px!important;position:absolute;right:-6px;top:-6px;height:16px;width:16px;line-height:16px;cursor:pointer;text-align:center;color:#d7d9e0}.new-pmd.bd_weixin_popup .c-tips-icon-close:active,.new-pmd.bd_weixin_popup .c-tips-icon-close:hover{color:#626675}.new-pmd .c-tools-share-tip-con{padding-bottom:0}.new-pmd .c-tools-favo-tip-con{padding-bottom:10px}.new-pmd .c-tools-favo-tip-con .favo-icon{background-image:url(https://pss.bdstatic.com/r/www/cache/static/protocol/https/global/img/favo_sprites_e33db52.png);background-repeat:no-repeat;height:16px;width:16px;background-size:32px 16px;display:inline-block;vertical-align:text-bottom}.new-pmd .c-tools-favo-tip-con .success-icon{background-position:0 0}.new-pmd .c-tools-favo-tip-con .fail-icon{background-position:-16px 0}.new-pmd .c-tools-tip-con{box-shadow:0 2px 10px 0 rgba(0,0,0,.1);border-radius:6px;border:0;font-size:13px!important;line-height:13px;padding:11px 10px 10px}.new-pmd .c-tools-tip-con h3{font-size:13px!important}.new-pmd .c-tools-tip-con a{text-decoration:none}.new-pmd .c-tools-tip-con .c-tip-menu li{margin-bottom:13px}.new-pmd .c-tools-tip-con .c-tip-menu li a{color:#333;line-height:13px;padding:0}.new-pmd .c-tools-tip-con .c-tip-menu li a:hover{color:#315efb;background:none!important;text-decoration:none}.new-pmd .c-tools-tip-con .c-tip-menu li a:active{color:#f73131}.new-pmd .c-tools-tip-con .c-tip-menu li:last-child{margin-bottom:0}.new-pmd .c-tools-tip-con .c-tip-menu ul{width:auto;padding:0}.new-pmd .c-tools-tip-con .c-tip-notice{width:164px;padding:0}.new-pmd .c-tools-tip-con .c-tip-notice .c-tip-notice-succ{color:#333;font-weight:400;padding-bottom:10px}.new-pmd .c-tools-tip-con .c-tip-notice .c-tip-item-succ:first-child{padding-bottom:8px}.new-pmd .c-tools-tip-con .c-tip-notice .c-tip-item-succ a{color:#2440b3}.new-pmd .c-tools-tip-con .c-tip-notice .c-tip-item-succ a:hover{text-decoration:underline;color:#315efb}.new-pmd .c-tools-tip-con .c-tip-notice .c-tip-item-succ a:active{color:#f73131}.new-pmd .c-tools-tip-con .c-tip-notice .c-tip-item-fail{color:#9195A3}.new-pmd .c-tools-tip-con .c-tip-notice .c-tip-item-fail a:hover{text-decoration:underline;color:#315efb}.new-pmd .c-tools-tip-con .c-tip-notice .c-tip-item-fail a:active{color:#f73131}.new-pmd .c-tools-tip-con .c-tips-icon-close{font-size:13px!important;width:13px;line-height:13px;color:#C4C7CE}.new-pmd .c-tools-tip-con .c-tips-icon-close:hover,.new-pmd .c-tools-tip-con .c-tips-icon-close:active{color:#626675}.new-pmd .c-tools-tip-con .c-tools-share{padding:0}.new-pmd .c-tools-tip-con .c-tools-share a:hover{color:#315efb}.new-pmd .c-tools-tip-con .c-tools-share a:active{color:#f73131}.new-pmd .c-tools-tip-con .c-tools-share .bds_v2_share_box{margin-right:0}.new-pmd .c-tools-tip-con .c-tip-arrow{top:-5px}.new-pmd .c-tools-tip-con .c-tip-arrow em{border-width:0 4px 5px;border-style:solid;border-color:transparent;border-bottom-color:#fff;box-shadow:0 2px 10px 0 rgba(0,0,0,.1)}.new-pmd .c-tools-tip-con .c-tip-arrow ins{display:none}body{min-width:1060px}.wrapper_new{font-family:Arial,sans-serif}.wrapper_new #head{border-bottom:0;min-width:1060px}.wrapper_new #head.s_down{box-shadow:0 2px 10px 0 rgba(0,0,0,.1)}.wrapper_new .s_form{height:70px;padding-left:16px}.wrapper_new #result_logo{margin:17px 0 0}.wrapper_new .fm{margin:15px 0 15px 16px}.wrapper_new .s_ipt_wr{width:590px;height:36px;border:2px solid #c4c7ce;border-radius:10px 0 0 10px;border-right:0;overflow:visible}.wrapper_new #form .s_ipt_wr.new-ipt-focus{border-color:#4e6ef2}.wrapper_new.wrapper_s .s_ipt_wr{width:478px}.wrapper_new .iptfocus.s_ipt_wr{border-color:#4e71f2!important}.wrapper_new .s_ipt_wr:hover{border-color:#A7AAB5}.wrapper_new .head_wrapper input{outline:0;-webkit-appearance:none}.wrapper_new .s_ipt{height:38px;font:16px/18px arial;padding:10px 0 10px 14px;margin:0;width:484px;background:transparent;border:0;outline:0;-webkit-appearance:none}.wrapper_new.wrapper_l .soutu-env-mac .has-voice #kw.s_ipt{width:471px}.wrapper_new.wrapper_s .soutu-env-mac .has-voice #kw.s_ipt{width:359px}.wrapper_new.wrapper_l .soutu-env-mac #kw.s_ipt,.wrapper_new.wrapper_l .soutu-env-nomac #kw.s_ipt{width:503px}.wrapper_new.wrapper_s .soutu-env-mac #kw.s_ipt,.wrapper_new.wrapper_s .soutu-env-nomac #kw.s_ipt{width:391px}.pc-fresh-wrapper .wrapper_new #head{min-width:calc(var(--wrapper-width) + var(--wrapper-padding) + 24px)}.wrapper_new .s_form.s_form_fresh{width:var(--wrapper-width);padding:0 calc(var(--container-left-gap) - 135px)}.wrapper_new .s_form.s_form_fresh .fm{margin-left:18px}.wrapper_new .s_form_fresh .s_ipt_wr{width:calc(var(--left-container-width) + 14px)}.wrapper_new.wrapper_s .s_form_fresh #form .bdsug-new,.wrapper_new .s_form_fresh #form .bdsug-new{width:calc(var(--left-container-width) + 14px)}.wrapper_new.wrapper_s .s_form_fresh .soutu-env-new .soutu-layer{width:calc(var(--left-container-width) + 16px + 102px)}.wrapper_new .s_form_fresh .soutu-env-new .soutu-layer .soutu-url-wrap,.wrapper_new.wrapper_s .s_form_fresh .soutu-env-new .soutu-layer .soutu-url-wrap,.wrapper_new .s_form_fresh .soutu-env-new .soutu-layer #soutu-url-kw,.wrapper_new.wrapper_s .s_form_fresh .soutu-env-new .soutu-layer #soutu-url-kw{width:calc(var(--left-container-width) + 16px)}.wrapper_new .s_form_fresh .soutu-env-new .soutu-layer .soutu-url-btn-new{width:102px}.wrapper_new .s_form_fresh .soutu-env-mac #kw.s_ipt,.wrapper_new .s_form_fresh .soutu-env-nomac #kw.s_ipt{width:577px}.wrapper_new .s_form_fresh .soutu-env-mac .has-voice #kw.s_ipt{width:545px}.wrapper_new .s_form_fresh .s_btn_wr{width:102px}.wrapper_new .s_form_fresh .s_btn_wr .s_btn{width:102px}.wrapper_new .s_ipt_tip{height:37px;line-height:35px}.wrapper_new .s_btn_wr{width:112px;position:relative;z-index:2;zoom:1;border:0}.wrapper_new .s_btn_wr .s_btn{cursor:pointer;width:112px;height:40px;line-height:41px;line-height:40px\9;background-color:#4e6ef2;border-radius:0 10px 10px 0;font-size:17px;font-family:Arial,sans-serif;box-shadow:none;font-weight:400;border:0;outline:0;letter-spacing:normal}.wrapper_new .s_btn_wr .s_btn:hover{background:#4662D9}.wrapper_new .ipt_rec,.wrapper_new .soutu-btn{background:url(https://pss.bdstatic.com/r/www/cache/static/protocol/https/global/img/nicon_10750f3.png) no-repeat;width:24px;height:20px}.wrapper_new .ipt_rec{background-position:0 -2px;top:50%;right:52px!important;margin-top:-10px}.wrapper_new .ipt_rec:hover{background:url(https://pss.bdstatic.com/r/www/cache/static/protocol/https/global/img/nicon_10750f3.png) no-repeat;background-position:0 -26px}.wrapper_new .ipt_rec:after{display:none}.wrapper_new .soutu-btn{background-position:0 -51px;right:16px;margin-top:-9px}.wrapper_new .soutu-btn:hover{background-position:0 -75px}@media only screen and (-webkit-min-device-pixel-ratio:2){.wrapper_new .soutu-btn,.wrapper_new .ipt_rec{background-image:url(https://pss.bdstatic.com/r/www/cache/static/protocol/https/global/img/nicon-2x_6258e1c.png);background-size:24px 96px}.wrapper_new .ipt_rec:hover{background-image:url(https://pss.bdstatic.com/r/www/cache/static/protocol/https/global/img/nicon-2x_6258e1c.png);background-size:24px 96px}}.wrapper_new #s_tab{color:#626675;padding-top:59px;background:0 0;padding-left:150px}.new_input_wrapper.wrapper_new #s_tab{padding-top:62px}.wrapper_new #s_tab a{color:#626675}.wrapper_new #s_tab .s-old-tag a,.wrapper_new #s_tab .s-old-tag b{width:auto;min-width:44px;margin-right:27px;line-height:28px;text-align:left;margin-top:4px}.wrapper_new #s_tab i{font-size:14px;font-weight:400}.wrapper_new #s_tab .cur-tab{color:#222;font-weight:400;border-bottom:0}.wrapper_new #s_tab .s-old-tag .cur-tab:before{font-family:cIconfont!important;color:#626675;margin-right:2px;content:&#39;\e608&#39;}.wrapper_new #s_tab .s-old-tag.s-peak-tag .cur-tab:before{color:#fff}.pc-fresh-wrapper .wrapper_new #s_tab .s-old-tag .cur-tab:after{height:1px;margin-top:0}.wrapper_new #s_tab .s-old-tag .cur-tab:after{content:&#39;&#39;;width:auto;min-width:44px;height:2px;background:#4e6ef2;border-radius:1px;display:block;margin-top:1px}.wrapper_new #s_tab .s-old-tag.s-peak-tag .cur-tab:after{background:#fff}.wrapper_new #s_tab .s-new-tag{display:flex;align-items:center}.wrapper_new #s_tab .s-new-tag a,.wrapper_new #s_tab .s-new-tag b{line-height:28px;margin-top:5px;display:inline-block;text-decoration:none;font-size:14px}.wrapper_new #s_tab .s-new-tag .s-tab-item{text-align:center;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.wrapper_new #s_tab .s-new-tag .cur-tab{margin-right:4px;text-align:left;width:50px;margin-right:-3px;position:relative;overflow:unset}.wrapper_new #s_tab .s-new-tag .cur-tab:after{content:&#39;&#39;;display:block;position:absolute;width:28px;min-width:28px;border-radius:1px;margin-top:1px}.wrapper_new #s_tab .s-new-tag .s-tab-item{width:44px;margin-right:18px}.wrapper_new #s_tab .s-new-tag .s-tab-csaitab{width:unset;margin-left:12px;margin-right:23px}.wrapper_new #s_tab .s-new-tag .s-tab-filter-divider{background-color:#b7b9c1;margin-top:5px}.wrapper_new #s_tab .s-new-tag .s-tab-filter{text-align:center;line-height:28px;margin-top:5px;margin-left:18px}.wrapper_new #s_tab .s-new-tag .s-tab-filter:hover{color:#000311;cursor:pointer}.wrapper_new #s_tab .s-new-tag.s-peak-tag .cur-tab:after{background:#fff}.wrapper_new.wrapper_s #s_tab .s-new-tag a,.wrapper_new.wrapper_s #s_tab .s-new-tag b{margin-right:15px}.wrapper_new.wrapper_s #s_tab .s-new-tag a.s-tab-csaitab{margin-left:5px;margin-right:12px}.wrapper_new.wrapper_s #s_tab .s-new-tag .cur-tab{width:36px}.wrapper_new.wrapper_s #s_tab .s-new-tag .s-tab-filter{margin-left:9px}.pc-fresh-wrapper .wrapper_new #s_tab,.pc-fresh-wrapper .wrapper_new.wrapper_s #s_tab,.pc-fresh-wrapper .wrapper_new.wrapper_l #s_tab{padding-left:0;padding-bottom:0;height:33px}.pc-fresh-wrapper .wrapper_new #s_tab .s_tab_inner.s-old-tag,.pc-fresh-wrapper .wrapper_new.wrapper_s #s_tab .s_tab_inner.s-old-tag,.pc-fresh-wrapper .wrapper_new.wrapper_l #s_tab .s_tab_inner.s-old-tag{display:flex;justify-content:space-between;box-sizing:content-box;width:var(--left-container-width);padding:0 var(--container-left-gap)}.pc-fresh-wrapper .wrapper_new #s_tab .s_tab_inner.s-old-tag b,.pc-fresh-wrapper .wrapper_new.wrapper_s #s_tab .s_tab_inner.s-old-tag b,.pc-fresh-wrapper .wrapper_new.wrapper_l #s_tab .s_tab_inner.s-old-tag b,.pc-fresh-wrapper .wrapper_new #s_tab .s_tab_inner.s-old-tag a,.pc-fresh-wrapper .wrapper_new.wrapper_s #s_tab .s_tab_inner.s-old-tag a,.pc-fresh-wrapper .wrapper_new.wrapper_l #s_tab .s_tab_inner.s-old-tag a{margin-right:0}.pc-fresh-wrapper .wrapper_new #s_tab .s_tab_inner.s-old-tag .s-tab-filter,.pc-fresh-wrapper .wrapper_new.wrapper_s #s_tab .s_tab_inner.s-old-tag .s-tab-filter,.pc-fresh-wrapper .wrapper_new.wrapper_l #s_tab .s_tab_inner.s-old-tag .s-tab-filter{color:#626675}.pc-fresh-wrapper .wrapper_new #s_tab .s_tab_inner.s-old-tag .s-tab-filter:hover,.pc-fresh-wrapper .wrapper_new.wrapper_s #s_tab .s_tab_inner.s-old-tag .s-tab-filter:hover,.pc-fresh-wrapper .wrapper_new.wrapper_l #s_tab .s_tab_inner.s-old-tag .s-tab-filter:hover{color:#000311;cursor:pointer}.pc-fresh-wrapper .wrapper_new #s_tab .s_tab_inner.s-old-tag .s-tab-filter.s-tab-filter-peak,.pc-fresh-wrapper .wrapper_new.wrapper_s #s_tab .s_tab_inner.s-old-tag .s-tab-filter.s-tab-filter-peak,.pc-fresh-wrapper .wrapper_new.wrapper_l #s_tab .s_tab_inner.s-old-tag .s-tab-filter.s-tab-filter-peak{color:#fff}.pc-fresh-wrapper .wrapper_new #s_tab .s_tab_inner.s-old-tag .s-tab-filter.s-tab-filter-peak:hover,.pc-fresh-wrapper .wrapper_new.wrapper_s #s_tab .s_tab_inner.s-old-tag .s-tab-filter.s-tab-filter-peak:hover,.pc-fresh-wrapper .wrapper_new.wrapper_l #s_tab .s_tab_inner.s-old-tag .s-tab-filter.s-tab-filter-peak:hover{color:#fff}#con-at [tpl=gaokao_pc_main] .main-header-right_7CqYF{overflow:hidden}.wrapper_new.wrapper_s #s_tab .s-old-tag a,.wrapper_new.wrapper_s #s_tab .s-old-tag b{margin-right:15px}.wrapper_new.wrapper_s #s_tab .s-new-tag a,.wrapper_new.wrapper_s #s_tab .s-new-tag b{margin-right:8px}.wrapper_new.wrapper_s #s_tab .s-new-tag .first-tab{width:36px}.wrapper_new.wrapper_s #s_tab .s-new-tag .s-tab-filter{margin-left:8px}.wrapper_new #s_tab .s-tab-item:hover{color:#222}.wrapper_new #s_tab .s-tab-item:hover:before{color:#626675}.wrapper_new #s_tab .s-tab-item:before{font-family:cIconfont!important;font-style:normal;-webkit-font-smoothing:antialiased;background:initial;margin-right:2px;color:#C0C2C8;display:inline-block}.wrapper_new #s_tab .s-old-tag .s-tab-news:before{content:&#39;\e606&#39;}.wrapper_new #s_tab .s-old-tag .s-tab-video:before{content:&#39;\e604&#39;}.wrapper_new #s_tab .s-old-tag .s-tab-pic:before{content:&#39;\e607&#39;}.wrapper_new #s_tab .s-old-tag .s-tab-zhidao:before{content:&#39;\e633&#39;}.wrapper_new #s_tab .s-old-tag .s-tab-wenku:before{content:&#39;\e605&#39;}.wrapper_new #s_tab .s-old-tag .s-tab-tieba:before{content:&#39;\e609&#39;}.wrapper_new #s_tab .s-old-tag .s-tab-b2b:before{content:&#39;\e603&#39;}.wrapper_new #s_tab .s-old-tag .s-tab-map:before{content:&#39;\e630&#39;}.wrapper_new #s_tab .s-old-tag .s-tab-realtime_ugc:before{content:&#39;\e689&#39;}.wrapper_new #s_tab .s-old-tag .s-tab-note:before{content:&#39;\e6ad&#39;}.wrapper_new #s_tab .pc_home_inner_tab .s-tab-csaitab{height:28px;background-size:14px 14px;background-image:url(https://psstatic.cdn.bcebos.com/basics/chat/AI_1721378175000.svg);background-repeat:no-repeat;background-position:left;padding-left:18px;cursor:pointer;min-width:auto}.wrapper_new #s_tab .pc_home_inner_tab .s-tab-csaitab:hover{background-image:url(https://psstatic.cdn.bcebos.com/basics/chat/logo_1721447231000.svg)}.wrapper_new #s_tab .pc_home_inner_tab .s-tab-item-img{min-width:auto}.wrapper_new #s_tab .pc_home_inner_tab .s-tab-item-img img{height:14px;margin-bottom:-2px}.wrapper_new #u{height:60px;margin:4px 0 0;padding-right:24px}.wrapper_new #u&gt;a{text-decoration:none;line-height:24px;font-size:13px;margin:20px 0 0 24px;display:inline-block;vertical-align:top;cursor:pointer;color:#222}.wrapper_new #u&gt;a:hover,#user .username:hover{text-decoration:none;color:#315efb}.wrapper_new #u .pf .c-icon-triangle-down{display:none}.wrapper_new #u .lb{color:#fff;background-color:#4e71f2;height:24px;width:48px;line-height:24px;border-radius:6px;display:inline-block;text-align:center;margin-top:18px}.wrapper_new #u .lb:hover{color:#fff}.wrapper_new #u #s-top-loginbtn,.wrapper_new #u #user{position:relative;display:inline-block}.wrapper_new #u #s-top-loginbtn a{margin-left:24px;margin-right:0}.wrapper_new #u #s-top-loginbtn a:hover{text-decoration:none}.wrapper_new #u .username{margin-left:21px;margin-top:15px;display:inline-block;height:30px}.wrapper_new #u .s-msg-count{display:none;margin-left:4px}.wrapper_new #u .s-top-img-wrapper{position:relative;width:28px;height:28px;border:1px solid #4e71f2;display:inline-block;border-radius:50%}.wrapper_new #u .s-top-img-wrapper img{padding:2px;width:24px;height:24px;border-radius:50%}.wrapper_new #u .s-top-username{display:inline-block;max-width:100px;overflow:hidden;white-space:nowrap;text-overflow:ellipsis;-o-text-overflow:ellipsis;vertical-align:top;margin-top:5px;margin-left:6px;font:13px/23px Arial,sans-serif}.wrapper_new #u .username .c-icon{display:none}#wrapper.wrapper_new .bdnuarrow{display:none}#wrapper.wrapper_new .bdpfmenu{display:none}#wrapper.wrapper_new .bdpfmenu,#wrapper.wrapper_new .usermenu{width:84px;padding:8px 0;background:#fff;box-shadow:0 2px 10px 0 rgba(0,0,0,.15);-webkit-box-shadow:0 2px 10px 0 rgba(0,0,0,.15);-moz-box-shadow:0 2px 10px 0 rgba(0,0,0,.15);-o-box-shadow:0 2px 10px 0 rgba(0,0,0,.15);border-radius:12px;*border:1px solid #d7d9e0;border:0;overflow:hidden}.wrapper_new .s-top-img-wrapper{display:none}.wrapper_new .bdpfmenu a,.wrapper_new .usermenu a{padding:3px 0 3px 16px;color:#333;font-size:13px;line-height:19px;width:52px;height:19px;border-radius:4px}.wrapper_new .bdpfmenu .first,.wrapper_new .usermenu .first{margin-top:2px}.wrapper_new .bdpfmenu .last,.wrapper_new .usermenu .last{margin-bottom:2px}.wrapper_new .bdpfmenu a:hover .set,.wrapper_new .usermenu a:hover .set{margin-left:-8px;margin-top:-1px}.wrapper_new #u .bdpfmenu a:hover,.wrapper_new #u .usermenu a:hover{color:#315efb;text-decoration:none;background:#F1F3FD;margin-left:8px}.wrapper_new #u .usermenu{display:none;width:84px;padding:8px 0;background:#fff;box-shadow:0 2px 10px 0 rgba(0,0,0,.15);border-radius:12px;*border:1px solid #d7d9e0;border:0;overflow:hidden;position:absolute}.wrapper_new #u .bdpfmenu a,.wrapper_new #u .usermenu a{background:#fff;color:#333;text-decoration:none;display:block;text-align:left;padding:3px 0 3px 16px;font-size:13px;line-height:19px;margin:2px 4px 4px 0}.wrapper_new #u .usermenu .logout:hover{cursor:pointer}.wrapper_new #form .bdsug-new{width:590px;top:31px;border-radius:0 0 10px 10px;border:2px solid #4e71f2!important;border-top:0!important;box-shadow:none;font-family:Arial,sans-serif;z-index:1}.wrapper_new.wrapper_s #form .bdsug-new{width:478px}.wrapper_new #form .bdsug-new ul{margin:7px 14px 0;padding:8px 0 7px;background:0 0;border-top:2px solid #f5f5f6}.wrapper_new #form .search-sug-bottom-tools{text-align:right;background:0 0;height:25px;line-height:15px;border-radius:0 0 10px 10px;padding-bottom:2px;margin-top:8px}.wrapper_new #form .search-sug-bottom-tools .setup_storeSug{margin-right:12px;text-decoration:none;cursor:pointer;font-size:13px;color:#9195a3}.wrapper_new #form .search-sug-bottom-tools .del_storeSug{margin-right:12px;text-decoration:none;cursor:pointer;font-size:13px;color:#9195a3}.wrapper_new #form .search-sug-bottom-tools .more_storeSug{color:#666;text-decoration:none;margin-right:12px;font-size:13px;color:#9195a3}.wrapper_new #form .search-sug-bottom-tools .bdsug-feedback{text-decoration:none;color:#9195A3;font-size:13px;cursor:pointer;margin-right:14px}.wrapper_new.wrapper_s #form .bdsug-new ul li{width:auto}.wrapper_new #form .bdsug-new ul li{position:relative;width:auto;padding-left:14px;margin-left:-14px;margin-right:-14px;color:#626675;line-height:28px;background:0 0;font-family:Arial,sans-serif}.wrapper_new #form .search-sug-short ul{width:342px}.wrapper_new #form .search-sug-short ul li{width:350px;padding-right:0}.wrapper_new #form .bdsug-new .search-history-title,.search-recommend-title{font-size:13px;color:#9195A3;line-height:28px;font-weight:400;height:28px;margin-left:22px}.wrapper_new #form .bdsug-new #sugSearchHisUl{width:350px;margin-right:0}.wrapper_new #form .bdsug-new #sugSearchHisUl .search-history-title{margin-bottom:8px}.wrapper_new #form .bdsug-new #sugSearchHisUl li{width:316px;padding-right:34px}.wrapper_new #form .bdsug-new #sugSearchHisUl li u{right:8px}.wrapper_s.wrapper_new #form .bdsug-new #sugSearchHisUl{width:254px}.wrapper_s.wrapper_new #form .bdsug-new #sugSearchHisUl li{width:205px}.wrapper_new #form .bdsug-new #leftSugSearchHisUl{position:absolute;top:0;left:0;width:100%;margin-right:0}.wrapper_new #form .bdsug-new #leftSugSearchHisUl .search-recommend-title{margin-bottom:8px;margin-left:0}.wrapper_new #form .bdsug-new #leftSugSearchHisUl li{width:auto;padding-right:34px}.wrapper_new #form .bdsug-new #leftSugSearchHisUl li u{right:8px}.wrapper_new #form .bdsug-new #sugSearchGuessUl{position:absolute;z-index:-1;top:0;right:0;width:226px;padding-left:8px;margin:0;margin-top:7px;margin:7px 14px 0;margin-left:17px}.wrapper_new #form .bdsug-new #sugSearchGuessUl ov .search-recommend-title{margin-left:22px;height:28px;line-height:28px}.wrapper_new #form .bdsug-new #sugSearchGuessUl li{width:218px;padding-left:8px;margin-left:14px;margin-right:0}.wrapper_new #form .bdsug-new #leftSugSearchGuessUl{position:absolute;top:0;width:226px;padding-left:8px;margin:0;margin-top:7px;margin:7px 14px 0;margin-left:17px}.wrapper_new #form .bdsug-new #leftSugSearchGuessUl .search-recommend-title{margin-left:22px;height:28px;line-height:28px}.wrapper_new #form .bdsug-new #leftSugSearchGuessUl li{width:218px;padding-left:8px;margin-left:14px;margin-right:0}.wrapper_new #form .bdsug-new #rightSugSearchHisUl{position:absolute;top:0;right:0;width:226px;padding-left:8px;margin:0;margin-top:7px}.wrapper_new #form .bdsug-new #rightSugSearchHisUl .search-history-title{margin-left:8px;height:28px;line-height:28px}.wrapper_new #form .bdsug-new #rightSugSearchHisUl li{padding-right:40px;width:auto;padding-left:8px;margin-left:0;margin-right:0}.wrapper_new #form .bdsug-sample3{overflow:hidden}.wrapper_new.wrapper_s #form .bdsug-sample3{overflow:hidden}.wrapper_new.wrapper_s #form .search-sug-short ul{width:254px}.wrapper_new.wrapper_s #form .search-sug-short ul li{width:238px}.wrapper_new #form .bdsug-new ul li .sug-search-icon,.wrapper_new #form .bdsug-new ul li .sug-history-icon{margin-right:4px;color:#222;line-height:16px;height:16px}.wrapper_new #form .bdsug-new ul li span{color:#626675}.wrapper_new #form .bdsug-new ul li b{font-weight:400;color:#222}.wrapper_new #form .bdsug-new .bdsug-store-del{font-size:13px;text-decoration:none;color:#9195A3;right:16px}.wrapper_new #form .bdsug-new .bdsug-store-del:hover{color:#315EFB;cursor:pointer}.wrapper_new #form .bdsug-new ul li:hover,.wrapper_new #form .bdsug-new ul li:hover span,.wrapper_new #form .bdsug-new ul li:hover b{cursor:pointer}#head .s-down #form .bdsug-new{top:32px}.wrapper_new #form .bdsug-new .bdsug-s,.wrapper_new #form .bdsug-new .bdsug-s span,.wrapper_new #form .bdsug-new .bdsug-s b{color:#315EFB}.wrapper_new #form .bdsug-new .bdsug-s{background-color:#F5F6F9!important}.wrapper_new #form .bdsug-new&gt;div span:hover,.wrapper_new #form .bdsug-new&gt;div a:hover{color:#315EFB!important}.wrapper_new #form #kw.new-ipt-focus{border-color:#4e6ef2}.wrapper_new .bdsug-new .bdsug-feedback-wrap{border-radius:0 0 10px 10px;background:0 0;line-height:19px;margin-bottom:3px;margin-top:-7px}.wrapper_new .bdsug-new .bdsug-feedback-wrap span{text-decoration:none;color:#9195A3;font-size:13px;cursor:pointer;margin-right:14px}.wrapper_new .bdsug-new .bdsug-feedback-wrap span:hover{color:#315EFB}.wrapper_new #form .bdsug-new ul li,.wrapper_new.new_input_wrapper .bdsug-new ul li{}.wrapper_new #form .bdsug-new ul li .sug-hot-orange,.wrapper_new.new_input_wrapper .bdsug-new ul li .sug-hot-orange,.wrapper_new #form .bdsug-new ul li .sug-hot-grey,.wrapper_new.new_input_wrapper .bdsug-new ul li .sug-hot-grey,.wrapper_new #form .bdsug-new ul li .sug-hot-blue,.wrapper_new.new_input_wrapper .bdsug-new ul li .sug-hot-blue{display:inline-block;width:12px;height:12px;font-size:12px;line-height:12px;padding:2px;text-align:center;font-weight:500;margin-left:6px;vertical-align:text-bottom}.wrapper_new #form .bdsug-new ul li .sug-hot-orange,.wrapper_new.new_input_wrapper .bdsug-new ul li .sug-hot-orange{display:inline-block;color:#fff;background:#F60;border-radius:4px}.wrapper_new #form .bdsug-new ul li .sug-new-tag,.wrapper_new.new_input_wrapper .bdsug-new ul li .sug-new-tag{text-align:center;margin-left:6px;box-sizing:border-box;font-size:12px;line-height:14px;padding:1px 4px;font-weight:500}.wrapper_new #form .bdsug-new ul li .sug-new-tag-grey,.wrapper_new.new_input_wrapper .bdsug-new ul li .sug-new-tag-grey{color:#858585;border:1px solid rgba(133,133,133,.5);border-radius:4px}.wrapper_new #form .bdsug-new ul li .sug-new-tag-blue,.wrapper_new.new_input_wrapper .bdsug-new ul li .sug-new-tag-blue{color:#36F;border:1px solid rgba(51,102,255,.4);border-radius:4px}.wrapper_new #form .bdsug-new ul li .sug-new-tag-orange,.wrapper_new.new_input_wrapper .bdsug-new ul li .sug-new-tag-orange{color:#F33;border:1px solid rgba(255,51,51,.4);border-radius:4px}.wrapper_new #form .bdsug-new ul li .sug-tag-text,.wrapper_new.new_input_wrapper .bdsug-new ul li .sug-tag-text{position:sticky;right:-4px;margin-top:-2.5px;display:inline-block;box-sizing:border-box;margin-left:6px;padding:1px;font-size:12px;line-height:12px;font-weight:500;text-align:center;border-radius:4px;vertical-align:middle;border-width:1px;border-style:solid}.wrapper_new #form .bdsug-new ul li .sug-tag-img,.wrapper_new.new_input_wrapper .bdsug-new ul li .sug-tag-img{height:18px;max-width:80px;vertical-align:middle;margin-left:6px}.wrapper_new .bdsug-new ul li .direct-sug-wrap{display:block;height:40px;padding:7px 0;color:#222}.wrapper_new .bdsug-new ul li .direct-sug-wrap:visited{color:#222}.wrapper_new .bdsug-new ul li .direct-sug-wrap .left-img-wrap{position:relative}.wrapper_new .bdsug-new ul li .direct-sug-wrap .left-img-wrap::before{content:&quot;&quot;;width:38px;height:38px;position:absolute;top:0;right:0;left:0;bottom:0;border:1px solid rgba(0,0,0,.06);border-radius:9px}.wrapper_new .bdsug-new ul li .direct-sug-wrap .left-img{box-sizing:border-box;float:left;width:40px;height:40px;border-radius:9px;margin-right:6px}.wrapper_new .bdsug-new ul li .direct-sug-wrap .content{float:left}.wrapper_new .bdsug-new ul li .direct-sug-wrap .header{margin-bottom:6px;line-height:18px;height:18px;vertical-align:middle}.wrapper_new .bdsug-new ul li .direct-sug-wrap .name{float:left;font-size:16px;color:#222;font-weight:500}.wrapper_new .bdsug-new ul li .direct-sug-wrap .tag-img{width:18px;margin-left:4px}.wrapper_new .bdsug-new ul li .direct-sug-wrap .more{line-height:14px;width:240px}.wrapper_new .bdsug-new ul li .direct-sug-wrap .more:visited{color:#222}.wrapper_new .bdsug-new ul li .direct-sug-wrap .brief,.wrapper_new .bdsug-new ul li .direct-sug-wrap .info{font-size:14px;color:#222;font-weight:400}.wrapper_new .bdsug-new ul li .direct-sug-wrap .brief{margin-right:6px}.wrapper_new .bdsug-new ul li .direct-sug-wrap .right-btn{float:right;margin-top:5px;margin-right:6px;width:60px;height:24px;text-align:center;font-size:14px;color:#36F;line-height:24px;background-image:linear-gradient(111deg,#e8f7ff 0,#edf0ff 100%);border-radius:12px}.wrapper_new .bdsug-new ul li .direct-sug-wrap .right-btn:hover{background:#315efb;color:#fff}.wrapper_new .bdsug-new ul li .bdsug-ai-guide{box-sizing:content-box;display:inline-flex;align-items:center;justify-content:center;position:absolute;top:5.5px;height:16px;background:#6E4BFA17;padding:0 3px;border-radius:3px;margin-left:7px}.wrapper_new .bdsug-new ul li .bdsug-ai-guide #bdsug-ai-text{color:#6E4BFA;font-size:12px}.wrapper_new .bdsug-new ul li .bdsug-logo{margin-right:3px;height:12px}.wrapper_new .bdsug-new ul li:hover .direct-sug-wrap .name{color:#315EFB}.wrapper_new .bdsug-new ul li:hover .direct-sug-wrap .brief{color:#222}.wrapper_new .soutu-env-new .soutu-layer{width:100%}.wrapper_new .soutu-env-new .soutu-layer .soutu-url-wrap,.wrapper_new .soutu-env-new .soutu-layer #soutu-url-kw{width:592px;height:40px}.wrapper_new.wrapper_s .soutu-env-new .soutu-layer{width:592px}.wrapper_new.wrapper_s.new_input_wrapper .soutu-env-new .soutu-layer{width:100%}.wrapper_new.wrapper_s.new_input_wrapper .soutu-env-new .soutu-layer .soutu-url-wrap,.wrapper_new.wrapper_s.new_input_wrapper .soutu-env-new .soutu-layer #soutu-url-kw{width:598px}.wrapper_new.wrapper_s .soutu-env-new .soutu-layer .soutu-url-wrap,.wrapper_new.wrapper_s .soutu-env-new .soutu-layer #soutu-url-kw{width:480px;height:40px}.wrapper_new .soutu-env-new .soutu-layer .soutu-url-btn-new{width:112px;height:40px;line-height:41px;line-height:40px\9}.wrapper_new .soutu-hover-tip,.wrapper_new .voice-hover{top:50px}.wrapper_new .bdlayer .c-icon{width:16px;height:100%;vertical-align:top}.wrapper_new #content_left{padding-left:140px}#content_left .search-source-wrap{position:relative;margin-top:-11px;margin-bottom:21px}#content_left .search-source-wrap .search-source-title{float:left;margin-right:8px;color:#626675}#content_left .search-source-wrap .search-source-content{display:inline-block}#content_left .search-source-wrap .search-source-content:hover .search-source-popup{display:block;background-color:#fff}#content_left .search-source-wrap .iconfont{font-family:cIconfont!important;font-style:normal;-webkit-font-smoothing:antialiased;-moz-osx-font-smoothing:grayscale;color:rgba(0,0,0,.1)}#content_left .search-source-wrap .iconfont:hover{cursor:pointer}#content_left .search-source-wrap .search-source-popup{display:none;position:absolute;z-index:1;box-sizing:border-box;left:69px;top:24px;padding:5.5px 10px;border-radius:6px;background-color:#fff;box-shadow:0 2px 10px 0 rgba(0,0,0,.1)}#content_left .search-source-wrap .search-source-popup .arrow{position:absolute;top:-5px;left:50%;width:0;height:0;margin-left:-4px;border-width:0 4px 5px;border-style:solid;border-color:transparent;border-bottom-color:#fff;box-shadow:0 2px 10px 0 rgba(0,0,0,.1)}#content_left .search-source-wrap .search-source-popup .feedback{text-align:center}#content_left .search-source-wrap .search-source-popup .feedback:hover{cursor:pointer;color:#315efb}.wrapper_new .search_tool_conter,.wrapper_new .nums,.wrapper_new #rs,.wrapper_new .hint_common_restop{margin-left:140px}.wrapper_new #rs{margin-bottom:10px}.wrapper_new #rs th{font-family:&#39;Microsoft YaHei&#39;,Arial,sans-serif}#help .activity{font-weight:700;text-decoration:underline}.index-logo-peak{display:none}.baozhang-new-v2{margin-left:8px}.c-trust-as.baozhang-new-v2 i{display:inline-block;vertical-align:text-bottom;font-family:none;width:43px;height:17px;background-image:url(https://pss.bdstatic.com/r/www/cache/static/protocol/https/global/img/bao_02f5d40.svg);background-repeat:no-repeat;background-size:contain;position:relative;top:1px}.c-trust-as.baozhang-new-v2+.c-trust-as.vstar a{position:relative;top:1px}@supports (-ms-ime-align:auto){.c-trust-as.baozhang-new-v2+.c-trust-as.vstar a{top:0}}#head_wrapper.s-down .soutu-env-new .soutu-layer #soutu-url-kw{height:40px!important}.body-brand{min-width:736px}.body-brand .wrapper_new #head,.body-brand .wrapper_new #s_tab,.body-brand .wrapper_new #top-ad{display:none}.body-brand .wrapper_new #foot .foot-inner #help{padding-left:48px!important;margin:0}.body-brand .wrapper_new #foot .foot-inner{margin:0}.body-brand .wrapper_new #wrapper_wrapper{margin-left:0}.body-brand .wrapper_new #wrapper_wrapper #container{margin-left:48px;padding-left:0;width:608px}.body-brand .wrapper_new #wrapper_wrapper #container .new_head_nums_cont_outer,.body-brand .wrapper_new #wrapper_wrapper #container #content_right{display:none}.body-brand .wrapper_new #wrapper_wrapper #container .brand-head{height:58px}.body-brand .wrapper_new #wrapper_wrapper #container .brand-head #result_logo{margin-top:14px}.body-brand .wrapper_new #wrapper_wrapper #container .brand-head .nums_text{color:#999;margin-top:19px;display:inline-block;margin-left:15px;width:auto}.body-brand .wrapper_new #wrapper_wrapper #container #rs{width:656px}.body-brand .wrapper_new #wrapper_wrapper #container #rs .new-inc-rs-table{width:656px}.body-brand .wrapper_new #wrapper_wrapper #container #rs .new-inc-rs-table th,.body-brand .wrapper_new #wrapper_wrapper #container #rs .new-inc-rs-table th .new-inc-rs-item{width:208px}.big-event-gray{filter:progid:DXImageTransform.Microsoft.BasicImage(grayscale=1);-webkit-filter:grayscale(100%);-moz-filter:grayscale(100%);-ms-filter:grayscale(100%);-o-filter:grayscale(100%);filter:grayscale(100%);filter:gray}.search-keyboard-toast{display:none;position:fixed;top:65px;left:134px;box-sizing:border-box;width:204px;height:40px;opacity:.9;background:#626675;border-radius:8px;color:#FFF;padding:11px 9px 10px 16px;font-size:13px}.search-keyboard-toast::before{position:absolute;left:47%;top:-11px;content:&quot; &quot;;width:0;height:0;border-width:6px;border-style:solid;border-color:transparent transparent #626675}.search-keyboard-toast .left-text{float:left}.search-keyboard-toast .left-text .key{text-align:center;display:inline-block;margin:0 3px 0 5px;width:12px;height:19px;line-height:19px;border:1px solid #fff;border-radius:2px}.search-keyboard-toast .right-icon{float:right;margin-top:3px}.search-keyboard-toast .right-icon:hover{cursor:pointer}.darkmode .search-keyboard-toast{background-color:#EEE;color:#333;opacity:.88}.darkmode .search-keyboard-toast::before{border-bottom-color:#EEE}.darkmode .search-keyboard-toast .left-text .key{border-color:#333}.new_input_wrapper.wrapper_new #s_tab{padding-top:62px}.pc-fresh-wrapper #content_right [tpl=right_game_recommend] .c-result-content{width:var(--right-container-width)}.pc-fresh-wrapper #content_right [tpl=right_game_recommend] .c-result-content .c-row .c-span8{width:var(--right-container-width)}.pc-fresh-wrapper #content_right [tpl=right_game_recommend] .c-result-content .c-row .c-span8 .c-span2{width:calc((var(--right-container-width) - var(--pc-col-gap) * 3) / 4)}.pc-fresh-wrapper #content_right [tpl=right_game_recommend] .c-result-content .c-row .c-span8 .c-span2 .c-img2{width:calc((var(--right-container-width) - var(--pc-col-gap) * 3) / 4);height:calc(((var(--right-container-width) - var(--pc-col-gap) * 3) / 4) * 1.25)}.pc-fresh-wrapper #content_right [tpl=right_game_recommend] .c-result-content .c-row .last-hidden{display:block}.pc-fresh-wrapper #content_right [tpl=right_game_recommend] .c-result-content .c-row .game-link{display:block;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.pc-fresh-wrapper #content_right [tpl=right_game_recommend] .c-result-content .c-row .item-desc{text-overflow:ellipsis;display:-webkit-box;-webkit-box-orient:vertical;-webkit-line-clamp:2}.pc-fresh-wrapper #content_right [tpl=right_game_recommend] .c-result-content .title-width{width:var(--right-container-width)}.pc-fresh-wrapper .c-container.result{width:var(--left-container-width)}.pc-fresh-wrapper .c-container.result .c-result-content header a:hover,.pc-fresh-wrapper .c-container.result .c-result-content article a:hover{text-decoration:none}#form.virtual-form{display:none}.new_input_wrapper .s_form{height:73px;transition:margin-left 160ms ease}#main-wrapper{width:710px;position:absolute;top:15px;left:135px;z-index:1}.wrapper_s #main-wrapper{width:592px}.wrapper_s #main-wrapper #main-input-wrapper #main-ipt{min-width:368px}#main-wrapper.two-line-sample{width:704px}.wrapper_s #main-wrapper.two-line-sample{width:592px}#main-wrapper.two-line-sample #main-input-wrapper .main-input.main-input-multi-line{padding-right:14px}#main-wrapper #main-input-wrapper{border-width:1.25px;border-color:#B7B9C1;border-style:solid;border-radius:12px;position:relative;transition:height 160ms ease;height:42px;background-color:#FFF;z-index:3}#main-wrapper #main-input-wrapper.height-line-1{height:42px}#main-wrapper #main-input-wrapper.height-line-2{height:83px}#main-wrapper #main-input-wrapper.height-line-3{height:105px!important}#main-wrapper #main-input-wrapper.height-line-4{height:127px!important}#main-wrapper #main-input-wrapper.height-line-5{height:149px!important}#main-wrapper #main-input-wrapper:hover{box-shadow:0 2px 12px 0 rgba(51,91,255,.08)}#main-wrapper #main-input-wrapper.wrapper-focus{border-color:#335BFF;box-shadow:0 2px 12px 0 rgba(51,91,255,.08)}#main-wrapper #main-input-wrapper .main-input{display:inline-block;min-width:482px;max-width:100%;word-wrap:break-word;max-height:88px;padding-left:16px;padding-right:14px;margin-right:4px;margin-top:12px;margin-bottom:12px;font-size:16px;line-height:18px;box-sizing:border-box;vertical-align:middle;overflow:hidden;outline:0;cursor:text;color:#333;caret-color:#335BFF}#main-wrapper #main-input-wrapper .main-input::-webkit-scrollbar{display:initial;width:4px}#main-wrapper #main-input-wrapper .main-input::-webkit-scrollbar-track{background:transparent}#main-wrapper #main-input-wrapper .main-input::-webkit-scrollbar-thumb{background:rgba(51,51,51,.2);border-radius:2px}#main-wrapper #main-input-wrapper .main-input.main-input-multi-line{line-height:22px;overflow-y:scroll;scroll-behavior:smooth;margin-top:10px;padding-right:16px}#main-wrapper #main-input-wrapper .main-input.main-input-blur{line-height:22px;overflow:hidden;white-space:nowrap;text-overflow:ellipsis;max-width:482px;margin-top:10px}.wrapper_s #main-wrapper #main-input-wrapper .main-input.main-input-blur{max-width:373px}#main-wrapper #main-input-wrapper .main-input .placeholder{color:#aaa}#main-wrapper #main-input-wrapper .main-submit{position:absolute;border-radius:0 12px 12px 0;right:-1px;bottom:-1px;height:44px;width:112px;background-color:#335bff;transition:all 160ms ease}#main-wrapper #main-input-wrapper .main-submit:before{content:&quot;&quot;;position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);background-image:url(https://gips0.baidu.com/it/u=3785693112,2605121048&amp;fm=3028&amp;app=3028&amp;f=PNG&amp;fmt=auto&amp;q=75&amp;size=f189_45);background-size:cover;width:63px;height:15px}#main-wrapper #main-input-wrapper .main-submit:hover{background-color:#2949cc;cursor:pointer}#main-wrapper #main-input-wrapper .main-submit.main-submit-multi-line{width:108px;height:36px;border-radius:9px;bottom:3px;right:3px}#main-wrapper #main-input-wrapper .main-tool{position:absolute;bottom:10px;right:112px;display:flex}#main-wrapper #main-input-wrapper .main-tool #pic-mic{display:flex}#main-wrapper #main-input-wrapper .main-tool #pic-mic .ipt_rec{position:relative;margin:0;height:21px;width:21px;margin-right:15px;top:unset;right:unset!important;background-image:url(https://pss.bdstatic.com/r/www/cache/static/protocol/https/global/img/voice_e255e65.png);background-size:21px 42px;background-repeat:no-repeat;background-position:0 0}#main-wrapper #main-input-wrapper .main-tool #pic-mic .ipt_rec:hover{cursor:pointer;background-position:0 -21px}#main-wrapper #main-input-wrapper .main-tool #pic-mic .soutu-btn{position:relative;display:block;margin:0;height:21px;width:21px;top:unset;right:unset;margin-right:17px;background-image:url(https://pss.bdstatic.com/r/www/cache/static/protocol/https/global/img/camera_4683ae6.png);background-size:21px 42px;background-repeat:no-repeat;background-position:0 0}#main-wrapper #main-input-wrapper .main-tool #pic-mic .soutu-btn:hover{cursor:pointer;background-position:0 -21px}#main-wrapper #main-input-wrapper .main-tool .clear-btn{display:none;width:21px;height:21px;margin-right:15px;background-image:url(https://pss.bdstatic.com/r/www/cache/static/protocol/https/global/img/clear_48eb6e3.png);background-size:21px 42px;background-repeat:no-repeat;background-position:0 0}#main-wrapper #main-input-wrapper .main-tool .clear-btn:hover{background-position:0 -21px;cursor:pointer}#main-wrapper #main-input-wrapper .voice-hover{right:40px;top:40px}#main-wrapper #main-input-wrapper .soutu-hover-tip{background:#fff;box-shadow:0 2px 10px 0 rgba(0,0,0,.1);border-radius:6px;height:32px;width:94px;text-align:center;line-height:32px;font-size:14px;color:#626675;position:absolute;z-index:2;top:40px;right:0}#main-wrapper.two-line-sample #main-input-wrapper.wrapper-focus{height:83px}#main-wrapper.two-line-sample #main-input-wrapper .main-input{overflow-y:scroll;max-width:100%}#main-wrapper.two-line-sample #main-input-wrapper .main-input.main-input-blur{max-width:375px;height:22px;white-space:nowrap;overflow-y:hidden}#main-wrapper.two-line-sample #main-input-wrapper .main-submit{width:108px;height:36px;border-radius:9px;bottom:3px;right:3px}#head .bdsug{top:0}#main-wrapper .bdsug-new{width:100%;border-radius:12px;position:relative;padding:10px;box-sizing:border-box;border:1px solid #dbdce0!important;box-shadow:0 2px 12px 0 rgba(51,91,255,.08);font-family:Arial,sans-serif;z-index:1;opacity:0;transform:translateY(-9px);transition:transform 160ms ease,opacity 160ms ease}.new_input_wrapper #main-wrapper .bdsug-new{width:100%}.new_input_wrapper #main-wrapper.two-line-sample .bdsug-new{}#main-wrapper .bdsug-new .bdsug-feedback-wrap{margin-top:0;height:unset;line-height:16px;margin-bottom:4px;margin-top:8px}#main-wrapper .bdsug-new .bdsug-feedback-wrap span{color:#626675;font-size:14px;margin-right:6px}#main-wrapper .bdsug-new .bdsug-feedback-wrap span:hover{color:#335BFF}#main-wrapper .bdsug-new ul{margin:unset;padding:unset;border-top:unset;background:unset}#main-wrapper .bdsug-new ul li{line-height:34px;font-size:16px;margin-left:0;margin-right:0;padding-left:5px;padding-right:0;width:100%;border-radius:9px;box-sizing:border-box;color:#9195A3}#main-wrapper .bdsug-new ul li .sug-tag-text{position:sticky;right:-4px;margin-top:-2.5px;display:inline-block;box-sizing:border-box;margin-left:6px;padding:1px;font-size:12px;line-height:12px;font-weight:500;text-align:center;border-radius:4px;vertical-align:middle;border-width:1px;border-style:solid}#main-wrapper .bdsug-new ul li .bdsug-store-del{font-size:14px;color:#626675;text-decoration:none;font-weight:400}#main-wrapper .bdsug-new ul li .bdsug-store-del:hover{color:#335BFF}#main-wrapper .bdsug-new ul span{color:#333}#main-wrapper .bdsug-new ul li b{font-weight:400;color:#333;margin-left:3px}#main-wrapper .bdsug-new ul li b.total-b{margin-left:0}#main-wrapper .bdsug-new ul li:hover{cursor:pointer;font-weight:500}#main-wrapper .bdsug-new ul li:hover b,#main-wrapper .bdsug-new ul li:hover span:not(.sug-tag-text){color:#335BFF;font-weight:500}#main-wrapper .bdsug-new .bdsug-feedback-tool-zone{line-height:16px!important;height:unset!important;padding-bottom:4px!important;margin-top:9px!important}#main-wrapper .bdsug-new .bdsug-feedback-tool-zone span{margin-right:15px!important;color:#626675!important}#main-wrapper .bdsug-new .bdsug-feedback-tool-zone a{color:#626675!important}#main-wrapper .bdsug-new .bdsug-feedback-tool-zone span:hover,#main-wrapper .bdsug-new .bdsug-feedback-tool-zone a:hover{color:#335BFF!important}#main-wrapper .bdsug-new .bdsug-feedback-tool-zone&gt;:last-child{margin-right:6px!important}#main-wrapper .bdsug-new ul li .direct-sug-wrap .right-btn{margin-top:7px}#main-wrapper .bdsug-new ul .bdsug-s,#main-wrapper .bdsug-new ul .bdsug-s b,#main-wrapper .bdsug-new ul .bdsug-s span:not(.sug-tag-text){color:#335BFF;background-color:#F5F6F9!important;font-weight:500}#main-wrapper .bdsug-new.enter{opacity:1;transform:translateY(9px)}#main-wrapper .bdsug-new.exit{opacity:0;transform:translateY(9px)}.wrapper_new.new_input_wrapper #u{margin-top:7px}.wrapper_new.new_input_wrapper #result_logo{margin:19.5px 0 0}.wrapper_new .s_form.sam_s_form{height:74px;padding-left:16px}.wrapper_new .sam_s_form#result_logo{margin:19px 0 0}.wrapper_new .sam_search.fm{margin:15px 0 15px 16px;border-radius:12px 14px 14px 12px}.wrapper_new .sam_search.fm:hover{border-color:#1d4fff}.wrapper_new .sam_search.fm:hover .s_btn_wr .s_btn{background:#1d4fff}.wrapper_new .sam_s_tab#s_tab{padding-top:63px!important}.wrapper_new .sam_search.sam_form_shadow{box-shadow:0 4px 2px 0 rgba(0,0,0,.1)}.wrapper_new .sam_search .s_ipt_wr{width:588px;height:40px;z-index:10;background-clip:padding-box;-ms-background-clip:padding-box;-webkit-background-clip:padding-box;border:2px solid #4e6ef2;border-radius:12px 10px 10px 12px;overflow:visible}.wrapper_new .sam_search .s_ipt_wr.ipthover{border-color:#1d4fff}.wrapper_new.wrapper_s .sam_search .s_ipt_wr{width:480px}.wrapper_new .sam_search .iptfocus.s_ipt_wr{border-color:#1d4fff!important}.wrapper_new .sam_search .s_ipt_wr:hover{border-color:#1d4fff}.wrapper_new .sam_search .s_ipt{height:40px;height:43px\0;font:18px/18px arial;padding:11px 0 11px 14px}.wrapper_new.wrapper_l .soutu-env-mac .sam_search.has-voice #kw.s_ipt{width:451px}.wrapper_new.wrapper_s .soutu-env-mac .sam_search.has-voice #kw.s_ipt{width:346px}.wrapper_new.wrapper_l .soutu-env-mac .sam_search #kw.s_ipt,.wrapper_new.wrapper_l .soutu-env-nomac .sam_search #kw.s_ipt{width:490px}.wrapper_new.wrapper_s .soutu-env-mac .sam_search #kw.s_ipt,.wrapper_new.wrapper_s .soutu-env-nomac .sam_search #kw.s_ipt{width:384px}.wrapper_new .sam_search .s_btn_wr{width:115px;margin-left:-9px;z-index:2;zoom:1;border:0}.wrapper_new .sam_search .s_btn_wr .s_btn{cursor:pointer;width:115px;height:44px;line-height:44px;padding-left:7px;padding-top:1px;background-color:#4e6ef2;border-radius:0 12px 12px 0;font-size:18px}.wrapper_new .sam_search .s_btn_wr .s_btn.btnfocus{background:#1d4fff}.wrapper_new .sam_search .s_btn_wr .s_btn:hover{background:#1d4fff}.wrapper_new .sam_search .s_btn_wr .s_btn:hover .wrapper_new .fm .s_ipt_wr{border-color:#1d4fff}.wrapper_new .sam_s_tab#s_tab{padding-top:63px!important}.wrapper_new .sam_s_form+#u{margin:7px 0 0}.wrapper_new #form .sam-bdsug.bdsug-new{width:100%;top:52px;border:1px solid rgba(0,0,0,.05)!important;box-shadow:0 4px 4px 0 rgba(0,0,0,.1);border-radius:12px}.wrapper_new.wrapper_s #form .sam-bdsug.bdsug-new{width:100%}.wrapper_new #form .sam-bdsug.bdsug-new ul{margin:6px 15px 0;padding:0 0 7px;background:0 0;border-top:0}.wrapper_new #form .sam-bdsug.bdsug-new ul li{width:auto;padding-left:15px;margin-left:-15px;margin-right:-15px;height:32px;line-height:32px}.wrapper_new #form .sam-bdsug.bdsug-new .bdsug-store-del{right:15px}.wrapper_new #form .sam-bdsug.bdsug-new .bdsug-s{background-color:#F1F3FD!important}.wrapper_new .sam-bdsug.bdsug-new .bdsug-feedback-wrap{margin-bottom:5px;margin-top:-3px}.wrapper_new .soutu-env-new .sam_search .soutu-layer{width:698px}.wrapper_new .soutu-env-new .soutu-layer .sam_url_wrap.soutu-url-wrap,.wrapper_new .soutu-env-new .soutu-layer #soutu-url-kw.sam_url_kw{width:588px;height:40px}.wrapper_new.wrapper_s .soutu-env-new .sam_search .soutu-layer{width:590px}.wrapper_new.wrapper_s .soutu-env-new .sam_url_wrap.soutu-url-wrap,.wrapper_new.wrapper_s .soutu-env-new #soutu-url-kw.sam_url_kw{width:480px;height:40px}.wrapper_new .soutu-env-new .soutu-layer .soutu-url-btn-new.sam_url_btn_new{width:114px;height:44px;line-height:44px;margin-left:-8px}.wrapper_new .soutu-env-new .soutu-layer .soutu-url-btn-new.sam_url_btn_new .sam_btn_text{display:inline-block;margin-left:6px;margin-top:1px}.head_wrapper .sam_search .sam_search_rec,.head_wrapper .sam_search .sam_search_soutu{z-index:1;display:none;position:absolute;top:50%;margin-top:-12px;font-size:24px;color:#4E6EF2;height:24px;line-height:24px;width:24px;cursor:pointer;-webkit-transform:translate3d(0,0,0);transform:translate3d(0,0,0);transition:transform .3s ease}.head_wrapper .sam_search .sam_search_rec{right:54px}.head_wrapper .sam_search .sam_search_soutu{right:14px}.head_wrapper .sam_search .sam_search_rec:hover,.head_wrapper .sam_search .sam_search_soutu:hover{color:#1D4FFF!important;transform:scale(1.08,1.08)}.head_wrapper .sam_search .sam_search_rec_hover,.head_wrapper .sam_search .sam_search_soutu_hover{background:#626675;border-radius:8px;height:32px;width:76px;text-align:center;line-height:32px;font-size:13px;color:#FFF;position:absolute;z-index:2;top:50px}.head_wrapper .sam_search .sam_search_rec_hover:before,.head_wrapper .sam_search .sam_search_soutu_hover:before{content:&#39;&#39;;border:4px solid transparent;border-bottom:4px solid #626675;position:absolute;left:50%;top:-8px;margin-left:-4px}.head_wrapper .sam_search .sam_search_rec_hover{right:29px}.head_wrapper .sam_search .sam_search_soutu_hover{display:none;right:-12px}.new_input_superman .soutu-env-new .soutu-layer{border-radius:12px;width:656px}.new_input_superman .soutu-env-new .soutu-layer .soutu-url{border:1.25px solid #335BFF;border-radius:12px;box-shadow:0 2px 12px 0 rgba(51,91,255,.08)}.new_input_superman .soutu-env-new .soutu-layer .soutu-url .soutu-url-wrap{border-radius:12px;height:42px}.new_input_superman .soutu-env-new .soutu-layer .soutu-url .soutu-url-wrap #soutu-url-kw{border:unset;width:100%;caret-color:#335BFF;font-weight:400;color:#333}.new_input_superman .soutu-env-new .soutu-layer .soutu-url .soutu-url-wrap #soutu-url-kw::-webkit-scrollbar{width:4px}.new_input_superman .soutu-env-new .soutu-layer .soutu-url .soutu-url-wrap #soutu-url-kw::placeholder{color:#9195A3}.new_input_superman .soutu-env-new .soutu-layer .soutu-url .soutu-url-wrap #soutu-url-kw::-webkit-input-placeholder{color:#9195A3}.new_input_superman .soutu-env-new .soutu-layer .soutu-url .soutu-url-wrap #soutu-url-kw::-moz-placeholder{color:#9195A3}.new_input_superman .soutu-env-new .soutu-layer .soutu-url .soutu-url-btn{height:42px;width:112px;background-color:#335bff;border-radius:0 12px 12px 0;position:absolute;right:-1px}.new_input_superman .soutu-env-new .soutu-layer .soutu-url .soutu-url-btn:before{content:&quot;&quot;;position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);background-image:url(https://gips0.baidu.com/it/u=3785693112,2605121048&amp;fm=3028&amp;app=3028&amp;f=PNG&amp;fmt=auto&amp;q=75&amp;size=f189_45);background-size:cover;width:63px;height:15px}.new_input_superman .soutu-env-new .soutu-layer .soutu-state-normal{margin-top:9px}.new_input_superman .soutu-env-new .soutu-layer.new-wrapper{width:100%}.new_input_superman .soutu-env-new .soutu-layer.new-wrapper .soutu-url .soutu-url-btn{position:absolute;bottom:3px;width:108px;height:36px;right:3px;border-radius:9px}.c-frame{margin-bottom:18px}.c-offset{padding-left:10px}.c-gray{color:#666}.c-gap-top-small{margin-top:5px}.c-gap-top{margin-top:10px}.c-gap-bottom-small{margin-bottom:5px}.c-gap-bottom{margin-bottom:10px}.c-gap-left{margin-left:12px}.c-gap-left-small{margin-left:6px}.c-gap-right{margin-right:12px}.c-gap-right-small{margin-right:6px}.c-gap-right-large{margin-right:16px}.c-gap-left-large{margin-left:16px}.c-gap-icon-right-small{margin-right:5px}.c-gap-icon-right{margin-right:10px}.c-gap-icon-left-small{margin-left:5px}.c-gap-icon-left{margin-left:10px}.c-container{width:538px;font-size:13px;line-height:1.54;word-wrap:break-word;word-break:break-word}.c-container .c-container{width:auto}.c-container table{border-collapse:collapse;border-spacing:0}.c-container td{font-size:13px;line-height:1.54}.c-default{font-size:13px;line-height:1.54;word-wrap:break-word;word-break:break-all}.c-container .t,.c-default .t{line-height:1.54}.c-default .t{margin-bottom:0}.cr-content{width:259px;font-size:13px;line-height:1.54;color:#333;word-wrap:break-word;word-break:normal}.cr-content table{border-collapse:collapse;border-spacing:0}.cr-content td{font-size:13px;line-height:1.54;vertical-align:top}.cr-offset{padding-left:17px}.cr-title{font-size:14px;line-height:1.29;font-weight:700}.cr-title-sub{float:right;font-size:13px;font-weight:400}.c-row{*zoom:1}.c-row:after{display:block;height:0;content:&quot;&quot;;clear:both;visibility:hidden}.c-span2{width:29px}.c-span3{width:52px}.c-span4{width:75px}.c-span5{width:98px}.c-span6{width:121px}.c-span7{width:144px}.c-span8{width:167px}.c-span9{width:190px}.c-span10{width:213px}.c-span11{width:236px}.c-span12{width:259px}.c-span13{width:282px}.c-span14{width:305px}.c-span15{width:328px}.c-span16{width:351px}.c-span17{width:374px}.c-span18{width:397px}.c-span19{width:420px}.c-span20{width:443px}.c-span21{width:466px}.c-span22{width:489px}.c-span23{width:512px}.c-span24{width:535px}.c-span2,.c-span3,.c-span4,.c-span5,.c-span6,.c-span7,.c-span8,.c-span9,.c-span10,.c-span11,.c-span12,.c-span13,.c-span14,.c-span15,.c-span16,.c-span17,.c-span18,.c-span19,.c-span20,.c-span21,.c-span22,.c-span23,.c-span24{float:left;_display:inline;margin-right:17px;list-style:none}.c-span-last{margin-right:0}.c-span-last-s{margin-right:0}.container_l .cr-content{width:351px}.container_l .cr-content .c-span-last-s{margin-right:17px}.container_l .cr-content-narrow{width:259px}.container_l .cr-content-narrow .c-span-last-s{margin-right:0}.c-border{width:518px;padding:9px;border:1px solid #e3e3e3;border-bottom-color:#e0e0e0;border-right-color:#ececec;box-shadow:1px 2px 1px rgba(0,0,0,.072);-webkit-box-shadow:1px 2px 1px rgba(0,0,0,.072);-moz-box-shadow:1px 2px 1px rgba(0,0,0,.072);-o-box-shadow:1px 2px 1px rgba(0,0,0,.072)}.c-border .c-gap-left{margin-left:10px}.c-border .c-gap-left-small{margin-left:5px}.c-border .c-gap-right{margin-right:10px}.c-border .c-gap-right-small{margin-right:5px}.c-border .c-border{width:auto;padding:0;border:0;box-shadow:none;-webkit-box-shadow:none;-moz-box-shadow:none;-o-box-shadow:none}.c-border .c-span2{width:34px}.c-border .c-span3{width:56px}.c-border .c-span4{width:78px}.c-border .c-span5{width:100px}.c-border .c-span6{width:122px}.c-border .c-span7{width:144px}.c-border .c-span8{width:166px}.c-border .c-span9{width:188px}.c-border .c-span10{width:210px}.c-border .c-span11{width:232px}.c-border .c-span12{width:254px}.c-border .c-span13{width:276px}.c-border .c-span14{width:298px}.c-border .c-span15{width:320px}.c-border .c-span16{width:342px}.c-border .c-span17{width:364px}.c-border .c-span18{width:386px}.c-border .c-span19{width:408px}.c-border .c-span20{width:430px}.c-border .c-span21{width:452px}.c-border .c-span22{width:474px}.c-border .c-span23{width:496px}.c-border .c-span24{width:518px}.c-border .c-span2,.c-border .c-span3,.c-border .c-span4,.c-border .c-span5,.c-border .c-span6,.c-border .c-span7,.c-border .c-span8,.c-border .c-span9,.c-border .c-span10,.c-border .c-span11,.c-border .c-span12,.c-border .c-span13,.c-border .c-span14,.c-border .c-span15,.c-border .c-span16,.c-border .c-span17,.c-border .c-span18,.c-border .c-span19,.c-border .c-span20,.c-border .c-span21,.c-border .c-span22,.c-border .c-span23,.c-border .c-span24{margin-right:10px}.c-border .c-span-last{margin-right:0}.c-loading{display:block;width:50px;height:50px;background:url(//www.baidu.com/aladdin/img/tools/loading.gif) no-repeat 0 0}.c-vline{display:inline-block;margin:0 3px;border-left:1px solid #ddd;width:0;height:12px;_vertical-align:middle;_overflow:hidden}.c-icon{background:url(https://pss.bdstatic.com/r/www/cache/static/protocol/https/global/img/icons_441e82f.png) no-repeat 0 0;_background-image:url(https://pss.bdstatic.com/r/www/cache/static/protocol/https/global/img/icons_d5b04cc.gif)}.c-icon{display:inline-block;width:14px;height:14px;vertical-align:text-bottom;font-style:normal;overflow:hidden}.c-icon-unfold,.c-icon-fold,.c-icon-chevron-unfold,.c-icon-chevron-fold{width:12px;height:12px}.c-icon-star,.c-icon-star-gray{width:60px}.c-icon-qa-empty,.c-icon-safeguard,.c-icon-register-empty,.c-icon-zan,.c-icon-music,.c-icon-music-gray,.c-icon-location,.c-icon-warning,.c-icon-doc,.c-icon-xls,.c-icon-ppt,.c-icon-pdf,.c-icon-txt,.c-icon-play-black,.c-icon-gift,.c-icon-baidu-share,.c-icon-bear,.c-icon-bear-border,.c-icon-location-blue,.c-icon-hotAirBall,.c-icon-moon,.c-icon-streetMap,.c-icon-mv,.c-icon-zhidao-s,.c-icon-shopping{width:16px;height:16px}.c-icon-bear-circle,.c-icon-warning-circle,.c-icon-warning-triangle,.c-icon-warning-circle-gray{width:18px;height:18px}.c-icon-tieba,.c-icon-zhidao,.c-icon-ball-blue,.c-icon-ball-red{width:38px;height:38px}.c-icon-unfold:hover,.c-icon-fold:hover,.c-icon-chevron-unfold:hover,.c-icon-chevron-fold:hover,.c-icon-download:hover,.c-icon-lyric:hover,.c-icon-v:hover,.c-icon-hui:hover,.c-icon-bao:hover,.c-icon-newbao:hover,.c-icon-person:hover,.c-icon-high-v:hover,.c-icon-phone:hover,.c-icon-nuo:hover,.c-icon-fan:hover,.c-icon-med:hover,.c-icon-air:hover,.c-icon-share2:hover,.c-icon-v1:hover,.c-icon-v2:hover,.c-icon-write:hover,.c-icon-R:hover{border-color:#388bff}.c-icon-unfold:active,.c-icon-fold:active,.c-icon-chevron-unfold:active,.c-icon-chevron-fold:active,.c-icon-download:active,.c-icon-lyric:active,.c-icon-v:active,.c-icon-hui:active,.c-icon-bao:active,.c-icon-newbao:active,.c-icon-person:active,.c-icon-high-v:active,.c-icon-phone:active,.c-icon-nuo:active,.c-icon-fan:active,.c-icon-med:active,.c-icon-air:active,.c-icon-share2:active,.c-icon-v1:active,.c-icon-v2:active,.c-icon-write:active,.c-icon-R:active{border-color:#a2a6ab;background-color:#f0f0f0;box-shadow:inset 1px 1px 1px #c7c7c7;-webkit-box-shadow:inset 1px 1px 1px #c7c7c7;-moz-box-shadow:inset 1px 1px 1px #c7c7c7;-o-box-shadow:inset 1px 1px 1px #c7c7c7}.c-icon-v3:hover{border-color:#ffb300}.c-icon-v3:active{border-color:#a2a6ab;background-color:#f0f0f0;box-shadow:inset 1px 1px 1px #c7c7c7;-webkit-box-shadow:inset 1px 1px 1px #c7c7c7;-moz-box-shadow:inset 1px 1px 1px #c7c7c7;-o-box-shadow:inset 1px 1px 1px #c7c7c7}.c-icon-unfold,.c-icon-fold,.c-icon-chevron-unfold,.c-icon-chevron-fold,.c-icon-download,.c-icon-lyric{border:1px solid #d8d8d8;cursor:pointer}.c-icon-v,.c-icon-hui,.c-icon-bao,.c-icon-newbao,.c-icon-person,.c-icon-high-v,.c-icon-phone,.c-icon-nuo,.c-icon-fan,.c-icon-med,.c-icon-air,.c-icon-share2,.c-icon-v1,.c-icon-v2,.c-icon-v3,.c-icon-write,.c-icon-R{border:1px solid #d8d8d8;cursor:pointer;border-color:transparent;_border-color:tomato;_filter:chroma(color=#ff6347)}.c-icon-v1,.c-icon-v2,.c-icon-v3,.c-icon-v1-noborder,.c-icon-v2-noborder,.c-icon-v3-noborder,.c-icon-v1-noborder-disable,.c-icon-v2-noborder-disable,.c-icon-v3-noborder-disable{width:19px}.c-icon-download,.c-icon-lyric{width:16px;height:16px}.c-icon-play-circle,.c-icon-stop-circle{width:18px;height:18px}.c-icon-play-circle-middle,.c-icon-stop-circle-middle{width:24px;height:24px}.c-icon-play-black-large,.c-icon-stop-black-large{width:36px;height:36px}.c-icon-play-black-larger,.c-icon-stop-black-larger{width:52px;height:52px}.c-icon-flag{background-position:0 -144px}.c-icon-bus{background-position:-24px -144px}.c-icon-calendar{background-position:-48px -144px}.c-icon-street{background-position:-72px -144px}.c-icon-map{background-position:-96px -144px}.c-icon-bag{background-position:-120px -144px}.c-icon-money{background-position:-144px -144px}.c-icon-game{background-position:-168px -144px}.c-icon-user{background-position:-192px -144px}.c-icon-globe{background-position:-216px -144px}.c-icon-lock{background-position:-240px -144px}.c-icon-plane{background-position:-264px -144px}.c-icon-list{background-position:-288px -144px}.c-icon-star-gray{background-position:-312px -144px}.c-icon-circle-gray{background-position:-384px -144px}.c-icon-triangle-down{background-position:-408px -144px}.c-icon-triangle-up{background-position:-432px -144px}.c-icon-triangle-up-empty{background-position:-456px -144px}.c-icon-sort-gray{background-position:-480px -144px}.c-icon-sort-up{background-position:-504px -144px}.c-icon-sort-down{background-position:-528px -144px}.c-icon-down-gray{background-position:-552px -144px}.c-icon-up-gray{background-position:-576px -144px}.c-icon-download-noborder{background-position:-600px -144px}.c-icon-lyric-noborder{background-position:-624px -144px}.c-icon-download-white{background-position:-648px -144px}.c-icon-close{background-position:-672px -144px}.c-icon-fail{background-position:-696px -144px}.c-icon-success{background-position:-720px -144px}.c-icon-triangle-down-g{background-position:-744px -144px}.c-icon-refresh{background-position:-768px -144px}.c-icon-chevron-left-gray{background-position:-816px -144px}.c-icon-chevron-right-gray{background-position:-840px -144px}.c-icon-setting{background-position:-864px -144px}.c-icon-close2{background-position:-888px -144px}.c-icon-chevron-top-gray-s{background-position:-912px -144px}.c-icon-fullscreen{background-position:0 -168px}.c-icon-safe{background-position:-24px -168px}.c-icon-exchange{background-position:-48px -168px}.c-icon-chevron-bottom{background-position:-72px -168px}.c-icon-chevron-top{background-position:-96px -168px}.c-icon-unfold{background-position:-120px -168px}.c-icon-fold{background-position:-144px -168px}.c-icon-chevron-unfold{background-position:-168px -168px}.c-icon-qa{background-position:-192px -168px}.c-icon-register{background-position:-216px -168px}.c-icon-star{background-position:-240px -168px}.c-icon-star-gray{position:relative}.c-icon-star-gray .c-icon-star{position:absolute;top:0;left:0}.c-icon-play-blue{background-position:-312px -168px}.c-icon-pic{width:16px;background-position:-336px -168px}.c-icon-chevron-fold{background-position:-360px -168px}.c-icon-video{width:18px;background-position:-384px -168px}.c-icon-circle-blue{background-position:-408px -168px}.c-icon-circle-yellow{background-position:-432px -168px}.c-icon-play-white{background-position:-456px -168px}.c-icon-triangle-down-blue{background-position:-480px -168px}.c-icon-chevron-unfold2{background-position:-504px -168px}.c-icon-right{background-position:-528px -168px}.c-icon-right-empty{background-position:-552px -168px}.c-icon-new-corner{width:15px;background-position:-576px -168px}.c-icon-horn{background-position:-600px -168px}.c-icon-right-large{width:18px;background-position:-624px -168px}.c-icon-wrong-large{background-position:-648px -168px}.c-icon-circle-blue-s{background-position:-672px -168px}.c-icon-play-gray{background-position:-696px -168px}.c-icon-up{background-position:-720px -168px}.c-icon-down{background-position:-744px -168px}.c-icon-stable{background-position:-768px -168px}.c-icon-calendar-blue{background-position:-792px -168px}.c-icon-triangle-down-blue2{background-position:-816px -168px}.c-icon-triangle-up-blue2{background-position:-840px -168px}.c-icon-down-blue{background-position:-864px -168px}.c-icon-up-blue{background-position:-888px -168px}.c-icon-ting{background-position:-912px -168px}.c-icon-piao{background-position:-936px -168px}.c-icon-wrong-empty{background-position:-960px -168px}.c-icon-warning-circle-s{background-position:-984px -168px}.c-icon-chevron-left{background-position:-1008px -168px}.c-icon-chevron-right{background-position:-1032px -168px}.c-icon-circle-gray-s{background-position:-1056px -168px}.c-icon-v,.c-icon-v-noborder{background-position:0 -192px}.c-icon-hui{background-position:-24px -192px}.c-icon-bao{background-position:-48px -192px}.c-icon-newbao{background-position:-97px -218px}.c-icon-phone{background-position:-72px -192px}.c-icon-qa-empty{background-position:-96px -192px}.c-icon-safeguard{background-position:-120px -192px}.c-icon-register-empty{background-position:-144px -192px}.c-icon-zan{background-position:-168px -192px}.c-icon-music{background-position:-192px -192px}.c-icon-music-gray{background-position:-216px -192px}.c-icon-location{background-position:-240px -192px}.c-icon-warning{background-position:-264px -192px}.c-icon-doc{background-position:-288px -192px}.c-icon-xls{background-position:-312px -192px}.c-icon-ppt{background-position:-336px -192px}.c-icon-pdf{background-position:-360px -192px}.c-icon-txt{background-position:-384px -192px}.c-icon-play-black{background-position:-408px -192px}.c-icon-play-black:hover{background-position:-432px -192px}.c-icon-gift{background-position:-456px -192px}.c-icon-baidu-share{background-position:-480px -192px}.c-icon-bear{background-position:-504px -192px}.c-icon-R{background-position:-528px -192px}.c-icon-bear-border{background-position:-576px -192px}.c-icon-person,.c-icon-person-noborder{background-position:-600px -192px}.c-icon-location-blue{background-position:-624px -192px}.c-icon-hotAirBall{background-position:-648px -192px}.c-icon-moon{background-position:-672px -192px}.c-icon-streetMap{background-position:-696px -192px}.c-icon-high-v,.c-icon-high-v-noborder{background-position:-720px -192px}.c-icon-nuo{background-position:-744px -192px}.c-icon-mv{background-position:-768px -192px}.c-icon-fan{background-position:-792px -192px}.c-icon-med{background-position:-816px -192px}.c-icon-air{background-position:-840px -192px}.c-icon-share2{background-position:-864px -192px}.c-icon-v1,.c-icon-v1-noborder{background-position:-888px -192px}.c-icon-v2,.c-icon-v2-noborder{background-position:-912px -192px}.c-icon-v3,.c-icon-v3-noborder{background-position:-936px -192px}.c-icon-v1-noborder-disable{background-position:-960px -192px}.c-icon-v2-noborder-disable{background-position:-984px -192px}.c-icon-v3-noborder-disable{background-position:-1008px -192px}.c-icon-write{background-position:-1032px -192px}.c-icon-zhidao-s{background-position:-1056px -192px}.c-icon-shopping{background-position:-1080px -192px}.c-icon-bear-circle{background-position:0 -216px}.c-icon-warning-circle{background-position:-24px -216px}.c-icon-warning-triangle{width:24px;background-position:-48px -216px}.c-icon-warning-circle-gray{background-position:-72px -216px}.c-icon-ball-red{background-position:0 -240px}.c-icon-ball-blue{background-position:-48px -240px}.c-icon-tieba{background-position:0 -288px}.c-icon-zhidao{background-position:-48px -288px}.c-icon-download{background-position:0 -336px}.c-icon-lyric{background-position:-24px -336px}.c-icon-play-circle{background-position:-48px -336px}.c-icon-play-circle:hover{background-position:-72px -336px}.c-icon-stop-circle{background-position:-96px -336px}.c-icon-stop-circle:hover{background-position:-120px -336px}.c-icon-play-circle-middle{background-position:0 -360px}.c-icon-play-circle-middle:hover{background-position:-48px -360px}.c-icon-stop-circle-middle{background-position:-96px -360px}.c-icon-stop-circle-middle:hover{background-position:-144px -360px}.c-icon-play-black-large{background-position:0 -408px}.c-icon-play-black-large:hover{background-position:-48px -408px}.c-icon-stop-black-large{background-position:-96px -408px}.c-icon-stop-black-large:hover{background-position:-144px -408px}.c-icon-play-black-larger{background-position:0 -456px}.c-icon-play-black-larger:hover{background-position:-72px -456px}.c-icon-stop-black-larger{background-position:-144px -456px}.c-icon-stop-black-larger:hover{background-position:-216px -456px}.c-recommend{font-size:0;padding:5px 0;border:1px solid #f3f3f3;border-left:0;border-right:0}.c-recommend .c-icon{margin-bottom:-4px}.c-recommend .c-gray,.c-recommend a{font-size:13px}.c-recommend-notopline{padding-top:0;border-top:0}.c-recommend-vline{display:inline-block;margin:0 10px -2px;border-left:1px solid #d8d8d8;width:0;height:12px;_vertical-align:middle;_overflow:hidden}.c-text{display:inline-block;padding:2px;text-align:center;vertical-align:text-bottom;font-size:12px;line-height:100%;font-style:normal;font-weight:400;color:#fff;overflow:hidden}a.c-text,a.c-text:hover,a.c-text:active,a.c-text:visited{color:#fff;text-decoration:none}.c-text-new{background-color:#f13f40}.c-text-info{padding-left:0;padding-right:0;font-weight:700;color:#2b99ff;*vertical-align:baseline;_position:relative;_top:2px}a.c-text-info,a.c-text-info:hover,a.c-text-info:active,a.c-text-info:visited{color:#2b99ff}.c-text-info b{_position:relative;_top:-1px}.c-text-info span{padding:0 2px;font-weight:400}.c-text-important{background-color:#1cb7fd}.c-text-public{background-color:#2b99ff}.c-text-warning{background-color:#ff830f}.c-text-prompt{background-color:#f5c537}.c-text-danger{background-color:#f13f40}.c-text-safe{background-color:#52c277}.c-text-empty{padding-top:1px;padding-bottom:1px;border:1px solid #d8d8d8;cursor:pointer;color:#23b9fd;background-color:#fff}a.c-text-empty,a.c-text-empty:visited{color:#23b9fd}.c-text-empty:hover{border-color:#388bff;color:#23b9fd}.c-text-empty:active{color:#23b9fd;border-color:#a2a6ab;background-color:#f0f0f0;box-shadow:inset 1px 1px 1px #c7c7c7;-webkit-box-shadow:inset 1px 1px 1px #c7c7c7;-moz-box-shadow:inset 1px 1px 1px #c7c7c7;-o-box-shadow:inset 1px 1px 1px #c7c7c7}.c-text-mult{padding-left:5px;padding-right:5px}.c-text-gray{background-color:#666}.c-btn,.c-btn:visited{color:#333!important}.c-btn{display:inline-block;padding:0 14px;margin:0;height:24px;line-height:25px;font-size:13px;filter:chroma(color=#000000);*zoom:1;border:1px solid #d8d8d8;cursor:pointer;font-family:inherit;font-weight:400;text-align:center;vertical-align:middle;background-color:#f9f9f9;overflow:hidden;outline:0}.c-btn:hover{border-color:#388bff}.c-btn:active{border-color:#a2a6ab;background-color:#f0f0f0;box-shadow:inset 1px 1px 1px #c7c7c7;-webkit-box-shadow:inset 1px 1px 1px #c7c7c7;-moz-box-shadow:inset 1px 1px 1px #c7c7c7;-o-box-shadow:inset 1px 1px 1px #c7c7c7}a.c-btn{text-decoration:none}button.c-btn{height:26px;_line-height:18px;*overflow:visible}button.c-btn::-moz-focus-inner{padding:0;border:0}.c-btn .c-icon{margin-top:5px}.c-btn-disable{color:#999!important}.c-btn-disable:visited{color:#999!important}.c-btn-disable:hover{border:1px solid #d8d8d8;cursor:default}.c-btn-disable:active{border-color:#d8d8d8;background-color:#f9f9f9;box-shadow:none;-webkit-box-shadow:none;-moz-box-shadow:none;-o-box-shadow:none}.c-btn-mini{padding-left:5px;padding-right:5px;height:18px;line-height:18px;font-size:12px}button.c-btn-mini{height:20px;_height:18px;_line-height:14px}.c-btn-mini .c-icon{margin-top:2px}.c-btn-large{height:28px;line-height:28px;font-size:14px;font-family:&quot;微软雅黑&quot;,&quot;黑体&quot;}button.c-btn-large{height:30px;_line-height:24px}.c-btn-large .c-icon{margin-top:7px;_margin-top:6px}.c-btn-primary,.c-btn-primary:visited{color:#fff!important}.c-btn-primary{background-color:#388bff;border-color:#3c8dff #408ffe #3680e6}.c-btn-primary:hover{border-color:#2678ec #2575e7 #1c6fe2 #2677e7;background-color:#388bff;background-image:url(data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAACCAMAAACuX0YVAAAABlBMVEVnpv85i/9PO5r4AAAAD0lEQVR42gEEAPv/AAAAAQAFAAIros7PAAAAAElFTkSuQmCC);*background-image:none;background-repeat:repeat-x;box-shadow:1px 1px 1px rgba(0,0,0,.4);-webkit-box-shadow:1px 1px 1px rgba(0,0,0,.4);-moz-box-shadow:1px 1px 1px rgba(0,0,0,.4);-o-box-shadow:1px 1px 1px rgba(0,0,0,.4)}.c-btn-primary:active{border-color:#178ee3 #1784d0 #177bbf #1780ca;background-color:#388bff;background-image:none;box-shadow:inset 1px 1px 1px rgba(0,0,0,.15);-webkit-box-shadow:inset 1px 1px 1px rgba(0,0,0,.15);-moz-box-shadow:inset 1px 1px 1px rgba(0,0,0,.15);-o-box-shadow:inset 1px 1px 1px rgba(0,0,0,.15)}.c-btn .c-icon{float:left}.c-dropdown2{position:relative;display:inline-block;width:100%;height:26px;line-height:26px;font-size:13px;vertical-align:middle;outline:0;_font-family:SimSun;background-color:#fff;word-wrap:normal;word-break:normal}.c-dropdown2 .c-dropdown2-btn-group{position:relative;height:24px;border:1px solid #999;border-bottom-color:#d8d8d8;border-right-color:#d8d8d8;-moz-user-select:none;-webkit-user-select:none;user-select:none}.c-dropdown2:hover .c-dropdown2-btn-group,.c-dropdown2-hover .c-dropdown2-btn-group{box-shadow:inset 1px 1px 0 0 #d8d8d8;-webkit-box-shadow:inset 1px 1px 0 0 #d8d8d8;-moz-box-shadow:inset 1px 1px 0 0 #d8d8d8;-o-box-shadow:inset 1px 1px 0 0 #d8d8d8}.c-dropdown2:hover .c-dropdown2-btn-icon,.c-dropdown2-hover .c-dropdown2-btn-icon{box-shadow:inset 0 1px 0 0 #d8d8d8;-webkit-box-shadow:inset 0 1px 0 0 #d8d8d8;-moz-box-shadow:inset 0 1px 0 0 #d8d8d8;-o-box-shadow:inset 0 1px 0 0 #d8d8d8}.c-dropdown2:hover .c-dropdown2-btn-icon-border,.c-dropdown2-hover .c-dropdown2-btn-icon-border{background-color:#f2f2f2}.c-dropdown2 .c-dropdown2-btn{height:24px;padding-left:10px;padding-right:10px;cursor:default;overflow:hidden;white-space:nowrap}.c-dropdown2 .c-dropdown2-btn-icon{position:absolute;top:0;right:0;width:23px;height:24px;line-height:24px;background-color:#fff;padding:0 1px 0 10px}.c-dropdown2 .c-dropdown2-btn-icon-border{height:24px;width:23px;border-left:1px solid #d9d9d9;text-align:center;zoom:1}.c-dropdown2 .c-icon-triangle-down{*margin-top:5px;_margin-left:2px}.c-dropdown2 .c-dropdown2-menu{position:absolute;left:0;top:100%;_margin-top:0;width:100%;overflow:hidden;border:1px solid #bbb;background:#fff;visibility:hidden}.c-dropdown2 .c-dropdown2-menu-inner{overflow:hidden}.c-dropdown2 .c-dropdown2-option{background-color:#fff;cursor:pointer}.c-dropdown2 .c-dropdown2-selected{background-color:#f5f5f5}.c-dropdown2-common ul,.c-dropdown2-common li{margin:0;padding:0;list-style:none}.c-dropdown2-common .c-dropdown2-option{height:26px;line-height:26px;font-size:12px;color:#333;white-space:nowrap;cursor:pointer;padding-left:10px}.c-dropdown2-common .c-dropdown2-selected{background-color:#f5f5f5}.c-dropdown2-common .c-dropdown2-menu-group .c-dropdown2-group{padding-left:10px;font-weight:700;cursor:default}.c-dropdown2-common .c-dropdown2-menu-group .c-dropdown2-option{padding-left:20px}.c-img{display:block;min-height:1px;border:0 0}.c-img3{width:52px}.c-img4{width:75px}.c-img6{width:121px}.c-img7{width:144px}.c-img12{width:259px}.c-img15{width:328px}.c-img18{width:397px}.c-border .c-img3{width:56px}.c-border .c-img4{width:78px}.c-border .c-img7{width:144px}.c-border .c-img12{width:254px}.c-border .c-img15{width:320px}.c-border .c-img18{width:386px}.c-index{display:inline-block;padding:1px 0;color:#fff;width:14px;line-height:100%;font-size:12px;text-align:center;background-color:#8eb9f5}.c-index-hot,.c-index-hot1{background-color:#f54545}.c-index-hot2{background-color:#ff8547}.c-index-hot3{background-color:#ffac38}.c-input{display:inline-block;padding:0 4px;height:24px;line-height:24px\9;font-size:13px;border:1px solid #999;border-bottom-color:#d8d8d8;border-right-color:#d8d8d8;outline:0;box-sizing:content-box;-webkit-box-sizing:content-box;-moz-box-sizing:content-box;vertical-align:top;overflow:hidden}.c-input:hover{box-shadow:inset 1px 1px 1px 0 #d8d8d8;-webkit-box-shadow:inset 1px 1px 1px 0 #d8d8d8;-moz-box-shadow:inset 1px 1px 1px 0 #d8d8d8;-o-box-shadow:inset 1px 1px 1px 0 #d8d8d8}.c-input .c-icon{float:right;margin-top:6px}.c-input .c-icon-left{float:left;margin-right:4px}.c-input input{float:left;height:22px;*padding-top:4px;margin-top:2px;font-size:13px;border:0;outline:0}.c-input{width:180px}.c-input input{width:162px}.c-input-xmini{width:65px}.c-input-xmini input{width:47px}.c-input-mini{width:88px}.c-input-mini input{width:70px}.c-input-small{width:157px}.c-input-small input{width:139px}.c-input-large{width:203px}.c-input-large input{width:185px}.c-input-xlarge{width:341px}.c-input-xlarge input{width:323px}.c-input12{width:249px}.c-input12 input{width:231px}.c-input20{width:433px}.c-input20 input{width:415px}.c-border .c-input{width:178px}.c-border .c-input input{width:160px}.c-border .c-input-xmini{width:68px}.c-border .c-input-xmini input{width:50px}.c-border .c-input-mini{width:90px}.c-border .c-input-mini input{width:72px}.c-border .c-input-small{width:156px}.c-border .c-input-small input{width:138px}.c-border .c-input-large{width:200px}.c-border .c-input-large input{width:182px}.c-border .c-input-xlarge{width:332px}.c-border .c-input-xlarge input{width:314px}.c-border .c-input12{width:244px}.c-border .c-input12 input{width:226px}.c-border .c-input20{width:420px}.c-border .c-input20 input{width:402px}.c-numberset{*zoom:1}.c-numberset:after{display:block;height:0;content:&quot;&quot;;clear:both;visibility:hidden}.c-numberset li{float:left;margin-right:17px;list-style:none}.c-numberset .c-numberset-last{margin-right:0}.c-numberset a{display:block;width:50px;text-decoration:none;text-align:center;border:1px solid #d8d8d8;cursor:pointer}.c-numberset a:hover{border-color:#388bff}.c-border .c-numberset li{margin-right:10px}.c-border .c-numberset .c-numberset-last{margin-right:0}.c-border .c-numberset a{width:54px}.c-table{width:100%;border-collapse:collapse;border-spacing:0}.c-table th,.c-table td{padding-left:10px;line-height:1.54;font-size:13px;border-bottom:1px solid #f3f3f3;text-align:left}.cr-content .c-table th:first-child,.cr-content .c-table td:first-child{padding-left:0}.c-table th{padding-top:4px;padding-bottom:4px;font-weight:400;color:#666;border-color:#f0f0f0;white-space:nowrap;background-color:#fafafa}.c-table td{padding-top:6.5px;padding-bottom:6.5px}.c-table-hasimg td{padding-top:10px;padding-bottom:10px}.c-table a,.c-table em{text-decoration:none}.c-table a:hover,.c-table a:hover em{text-decoration:underline}.c-table a.c-icon:hover{text-decoration:none}.c-table .c-btn:hover,.c-table .c-btn:hover em{text-decoration:none}.c-table-nohihead th{background-color:transparent}.c-table-noborder td{border-bottom:0}.c-tabs-nav-movetop{margin:-10px -9px 0 -10px;position:relative}.c-tabs-nav{border-bottom:1px solid #d9d9d9;background-color:#fafafa;line-height:1.54;font-size:0;*zoom:1;_overflow-x:hidden;_position:relative}.c-tabs-nav:after{display:block;height:0;content:&quot;&quot;;clear:both;visibility:hidden}.c-tabs-nav .c-tabs-nav-btn{float:right;_position:absolute;_top:0;_right:0;_z-index:1;background:#fafafa}.c-tabs-nav .c-tabs-nav-btn .c-tabs-nav-btn-prev,.c-tabs-nav .c-tabs-nav-btn .c-tabs-nav-btn-next{float:left;padding:6px 2px;cursor:pointer}.c-tabs-nav .c-tabs-nav-btn .c-tabs-nav-btn-disable{cursor:default}.c-tabs-nav .c-tabs-nav-view{_position:relative;overflow:hidden;*zoom:1;margin-bottom:-1px}.c-tabs-nav .c-tabs-nav-view .c-tabs-nav-li{margin-bottom:0}.c-tabs-nav .c-tabs-nav-more{float:left;white-space:nowrap}.c-tabs-nav li,.c-tabs-nav a{color:#666;font-size:13px;*zoom:1}.c-tabs-nav li{display:inline-block;margin-bottom:-1px;*display:inline;padding:3px 15px;vertical-align:bottom;border-style:solid;border-width:2px 1px 0;border-color:transparent;_border-color:tomato;_filter:chroma(color=#ff6347);list-style:none;cursor:pointer;white-space:nowrap;overflow:hidden}.c-tabs-nav a{text-decoration:none}.c-tabs-nav .c-tabs-nav-sep{height:16px;width:0;padding:0;margin-bottom:4px;border-style:solid;border-width:0 1px;border-color:transparent #fff transparent #dedede}.c-tabs-nav .c-tabs-nav-selected{_position:relative;border-color:#2c99ff #e4e4e4 #fff #dedede;background-color:#fff;color:#000;cursor:default}.c-tabs-nav-one .c-tabs-nav-selected{border-color:transparent;_border-color:tomato;_filter:chroma(color=#ff6347);background-color:transparent;color:#666}.c-tabs .c-tabs .c-tabs-nav{padding:10px 0 5px;border:0 0;background-color:#fff}.c-tabs .c-tabs .c-tabs-nav li,.c-tabs .c-tabs .c-tabs-nav a{color:#00c}.c-tabs .c-tabs .c-tabs-nav li{padding:0 5px;position:static;margin:0 10px;border:0 0;cursor:pointer;white-space:nowrap}.c-tabs .c-tabs .c-tabs-nav .c-tabs-nav-sep{height:11px;width:0;padding:0;margin:0 0 4px;border:0 0;border-left:1px solid #d8d8d8}.c-tabs .c-tabs .c-tabs-nav .c-tabs-nav-selected{background-color:#2c99ff;color:#fff;cursor:default}.c-tag{padding-top:3px;margin-bottom:3px;height:1.7em;font-size:13px;line-height:1.4em;transition:height .3s ease-in;-webkit-transition:height .3s ease-in;-moz-transition:height .3s ease-in;-ms-transition:height .3s ease-in;-o-transition:height .3s ease-in;*zoom:1;overflow:hidden}.c-tag:after{display:block;height:0;content:&quot;&quot;;clear:both;visibility:hidden}.c-tag-cont{overflow:hidden;*zoom:1}.c-tag-type,.c-tag-li,.c-tag-more,.c-tag-cont span{margin:2px 0}.c-tag-type,.c-tag-li,.c-tag-cont span{float:left}.c-tag-type,.c-tag-more{color:#666}.c-tag-li,.c-tag-cont span{padding:0 4px;display:inline-block;margin-right:12px;white-space:nowrap;cursor:pointer;color:#00c}.c-tag .c-tag-selected{background:#388bff;color:#fff}.c-tag-more{float:right;background:#fff;cursor:pointer;*height:18px}.c-tool{display:inline-block;width:56px;height:56px;background:url(//www.baidu.com/aladdin/img/tools/tools-5.png) no-repeat}.c-tool-region{background-position:0 0}.c-tool-calendar{background-position:-72px 0}.c-tool-city{background-position:-144px 0}.c-tool-phone-pos{background-position:-216px 0}.c-tool-other{background-position:-288px 0}.c-tool-midnight{background-position:-360px 0}.c-tool-kefu{width:121px;background-position:-432px 0}.c-tool-phone{background-position:-576px 0}.c-tool-car{background-position:-648px 0}.c-tool-station{background-position:0 -72px}.c-tool-cheat{background-position:-72px -72px}.c-tool-counter{background-position:-144px -72px}.c-tool-time{background-position:-216px -72px}.c-tool-zip{background-position:-288px -72px}.c-tool-warning{background-position:-360px -72px}.c-tool-ip{background-position:0 -144px}.c-tool-unit{background-position:-72px -144px}.c-tool-rate{background-position:-144px -144px}.c-tool-conversion{background-position:-288px -144px}.c-tool-ads{background-position:-360px -144px}.c-icon-baozhang-new{width:14px;height:14px;background-image:url(https://pss.bdstatic.com/r/www/cache/static/protocol/https/global/img/pc-bao_96f4fc0.png);background-size:140px 14px;background-repeat:no-repeat;cursor:pointer;border-color:transparent;margin-left:11px;margin-right:3px}.c-icon-baozhang-new.animate{-webkit-animation-name:keyframesBao;animation-name:keyframesBao;-webkit-animation-duration:1s;animation-duration:1s;-webkit-animation-delay:0s;animation-delay:0s;-webkit-animation-iteration-count:1;animation-iteration-count:1;-webkit-animation-fill-mode:forwards;animation-fill-mode:forwards;-webkit-animation-timing-function:steps(1);animation-timing-function:steps(1)}@-webkit-keyframes keyframesBao{0%{background-position:0 0}10%{background-position:-14px 0}20%{background-position:-28px 0}30%{background-position:-42px 0}40%{background-position:-56px 0}50%{background-position:-70px 0}60%{background-position:-84px 0}70%{background-position:-98px 0}80%{background-position:-112px 0}90%,100%{background-position:-126px 0}}@keyframes keyframesBao{0%{background-position:0 0}10%{background-position:-14px 0}20%{background-position:-28px 0}30%{background-position:-42px 0}40%{background-position:-56px 0}50%{background-position:-70px 0}60%{background-position:-84px 0}70%{background-position:-98px 0}80%{background-position:-112px 0}90%,100%{background-position:-126px 0}}.opui-honourCard4-new-bao-title{font-size:12px;line-height:16px;color:#333;margin:3px 10px 0}.c-tip-con .opui-honourCard4-new-bao-style{width:100%;margin-top:4px}.c-tip-con .opui-honourCard4-new-bao-style a,.c-tip-con .opui-honourCard4-new-bao-style a:visited{color:#666}.new-pmd{}.new-pmd .c-gap-top-small{margin-top:6px}.new-pmd .c-gap-top{margin-top:8px}.new-pmd .c-gap-top-large{margin-top:12px}.new-pmd .c-gap-top-mini{margin-top:2px}.new-pmd .c-gap-top-xsmall{margin-top:4px}.new-pmd .c-gap-top-middle{margin-top:10px}.new-pmd .c-gap-bottom-small{margin-bottom:6px}.new-pmd .c-gap-bottom{margin-bottom:8px}.new-pmd .c-gap-bottom-large{margin-bottom:12px}.new-pmd .c-gap-bottom-mini{margin-bottom:2px}.new-pmd .c-gap-bottom-xsmall{margin-bottom:4px}.new-pmd .c-gap-bottom-middle{margin-bottom:10px}.new-pmd .c-gap-left{margin-left:12px}.new-pmd .c-gap-left-small{margin-left:8px}.new-pmd .c-gap-left-xsmall{margin-left:4px}.new-pmd .c-gap-left-mini{margin-left:2px}.new-pmd .c-gap-left-large{margin-left:16px}.new-pmd .c-gap-left-middle{margin-left:10px}.new-pmd .c-gap-right{margin-right:12px}.new-pmd .c-gap-right-small{margin-right:8px}.new-pmd .c-gap-right-xsmall{margin-right:4px}.new-pmd .c-gap-right-mini{margin-right:2px}.new-pmd .c-gap-right-large{margin-right:16px}.new-pmd .c-gap-right-middle{margin-right:10px}.new-pmd .c-gap-icon-right-small{margin-right:5px}.new-pmd .c-gap-icon-right{margin-right:10px}.new-pmd .c-gap-icon-left-small{margin-left:5px}.new-pmd .c-gap-icon-left{margin-left:10px}.new-pmd .c-row{*zoom:1}.new-pmd .c-row:after{display:block;height:0;content:&quot;&quot;;clear:both;visibility:hidden}.new-pmd .c-span1{width:32px}.new-pmd .c-span2{width:80px}.new-pmd .c-span3{width:128px}.new-pmd .c-span4{width:176px}.new-pmd .c-span5{width:224px}.new-pmd .c-span6{width:272px}.new-pmd .c-span7{width:320px}.new-pmd .c-span8{width:368px}.new-pmd .c-span9{width:416px}.new-pmd .c-span10{width:464px}.new-pmd .c-span11{width:512px}.new-pmd .c-span12{width:560px}.new-pmd .c-span2,.new-pmd .c-span3,.new-pmd .c-span4,.new-pmd .c-span5,.new-pmd .c-span6,.new-pmd .c-span7,.new-pmd .c-span8,.new-pmd .c-span9,.new-pmd .c-span10,.new-pmd .c-span11,.new-pmd .c-span12{float:left;_display:inline;margin-right:16px;list-style:none}.new-pmd .c-span-last{margin-right:0}.new-pmd .c-span-last-s{margin-right:0}.new-pmd .c-icon{font-family:cIconfont!important;font-style:normal;-webkit-font-smoothing:antialiased;-moz-osx-font-smoothing:grayscale}.new-pmd .c-index{display:inline-block;width:14px;padding:1px 0;line-height:100%;text-align:center;color:#fff;background-color:#8eb9f5;font-size:12px}.new-pmd .c-index-hot,.new-pmd .c-index-hot1{background-color:#f54545}.new-pmd .c-index-hot2{background-color:#ff8547}.new-pmd .c-index-hot3{background-color:#ffac38}.new-pmd .c-index-single{display:inline-block;background:0 0;color:#9195A3;width:18px;font-size:15px;letter-spacing:-1px}.new-pmd .c-index-single-hot,.new-pmd .c-index-single-hot1{color:#FE2D46}.new-pmd .c-index-single-hot2{color:#F60}.new-pmd .c-index-single-hot3{color:#FAA90E}.new-pmd .c-text{display:inline-block;padding:0 2px;text-align:center;vertical-align:middle;font-style:normal;color:#fff;overflow:hidden;line-height:16px;height:16px;font-size:12px;border-radius:4px;font-weight:400}.new-pmd a.c-text{text-decoration:none!important}.new-pmd .c-text-info{padding-left:0;padding-right:0;font-weight:700;color:#2b99ff;vertical-align:text-bottom}.new-pmd .c-text-info span{padding:0 2px;font-weight:400}.new-pmd .c-text-important{background-color:#1cb7fd}.new-pmd .c-text-public{background-color:#4E6EF2}.new-pmd .c-text-warning{background-color:#f60}.new-pmd .c-text-prompt{background-color:#ffc20d}.new-pmd .c-text-danger{background-color:#f73131}.new-pmd .c-text-safe{background-color:#39b362}.new-pmd .c-text-mult{padding:0 4px;line-height:18px;height:18px;border-radius:4px;font-weight:400}.new-pmd .c-text-blue{background-color:#4E6EF2}.new-pmd .c-text-blue-border{border:1px solid #CBD2FF;padding:0 8px;border-radius:4px;font-weight:400;color:#4E6EF2!important}.new-pmd .c-text-green{background-color:#39b362}.new-pmd .c-text-green-border{border:1px solid #C9E7CD;padding:0 8px;border-radius:4px;font-weight:400;color:#39b362!important}.new-pmd .c-text-red{background-color:#f73131}.new-pmd .c-text-red-border{border:1px solid #F0C8BD;padding:0 8px;border-radius:4px;font-weight:400;color:#f73131!important}.new-pmd .c-text-yellow{background-color:#ffc20d}.new-pmd .c-text-yellow-border{border:1px solid #FCEDB1;padding:0 8px;border-radius:4px;font-weight:400;color:#ffc20d!important}.new-pmd .c-text-orange{background-color:#f60}.new-pmd .c-text-orange-border{border:1px solid #F8D2B0;padding:0 8px;border-radius:4px;font-weight:400;color:#f60!important}.new-pmd .c-text-pink{background-color:#fc3274}.new-pmd .c-text-pink-border{border:1px solid #F6C4D7;padding:0 8px;border-radius:4px;font-weight:400;color:#fc3274!important}.new-pmd .c-text-gray{background-color:#626675}.new-pmd .c-text-gray-border{border:1px solid #DBDBDB;padding:0 8px;border-radius:4px;font-weight:400;color:#626675!important}.new-pmd .c-text-dark-red{background-color:#CC2929}.new-pmd .c-text-gray-opacity{background-color:rgba(0,0,0,.3)}.new-pmd .c-text-white-border{border:1px solid rgba(255,255,255,.8);padding:0 8px;border-radius:4px;font-weight:400;color:#fff!important}.new-pmd .c-text-hot{background-color:#F60}.new-pmd .c-text-new{background-color:#FF455B}.new-pmd .c-text-fei{background-color:#FC3200}.new-pmd .c-text-bao{background-color:#DE1544}.new-pmd .c-text-rec{background-color:#4DADFE}.new-pmd .c-text-business{background-color:#8399F5}.new-pmd .c-text-time{background-color:rgba(0,0,0,.3)}.new-pmd .c-text-free-download{border:1px solid rgba(58,179,98,.5);padding:0 8px;border-radius:4px;font-weight:400;color:#3AB362!important;padding:0 5px;border-radius:8px;background-color:rgba(58,179,98,.1)}.new-pmd .c-text-authoritative{height:16px;line-height:16px;font-size:12px;font-weight:500;padding:1px 6px;border-radius:5px;color:#A95212;background-color:rgba(253,179,63,.12)}.new-pmd .c-btn,.new-pmd .c-btn:visited{color:#333!important}.new-pmd .c-btn{display:inline-block;overflow:hidden;font-family:inherit;font-weight:400;text-align:center;vertical-align:middle;outline:0;border:0;height:30px;width:80px;line-height:30px;font-size:13px;border-radius:6px;padding:0;background-color:#F5F5F6;*zoom:1;cursor:pointer}.new-pmd a.c-btn{text-decoration:none}.new-pmd button.c-btn{*overflow:visible;border:0;outline:0}.new-pmd button.c-btn::-moz-focus-inner{padding:0;border:0}.new-pmd .c-btn-disable{color:#C4C7CE!important}.new-pmd .c-btn-disable:visited{color:#C4C7CE!important}.new-pmd .c-btn-disable:hover{cursor:default;color:#C4C7CE!important;background-color:#F5F5F6}.new-pmd .c-btn-mini{height:24px;width:48px;line-height:24px}.new-pmd .c-btn-mini .c-icon{margin-top:2px}.new-pmd .c-btn-large{height:30px;line-height:30px;font-size:14px}.new-pmd button.c-btn-large{height:30px}.new-pmd .c-btn-large .c-icon{margin-top:7px}.new-pmd .c-btn-primary,.new-pmd .c-btn-primary:visited{color:#fff!important}.new-pmd .c-btn-primary{background-color:#4E6EF2}.new-pmd .c-btn-primary:hover{background-color:#315EFB;border:0!important;box-shadow:none!important;background-image:none!important}.new-pmd .c-btn-primary:active{border:0!important;box-shadow:none!important;background-image:none!important}.new-pmd .c-btn-default:hover{background-color:#315EFB;color:#FFF!important}.new-pmd .c-btn-weak{height:24px;line-height:24px;border-radius:4px;font-size:12px}.new-pmd .c-btn-add{width:32px;height:32px;line-height:32px;text-align:center;color:#9195A3!important}.new-pmd .c-btn-add:hover{background-color:#4E6EF2;color:#fff!important}.new-pmd .c-btn-add .c-icon{float:none}.new-pmd .c-btn-add-disable:hover{cursor:default;color:#C4C7CE!important;background-color:#F5F5F6}.new-pmd .c-tag{color:#333;display:inline-block;padding:0 8px;height:30px;line-height:30px;font-size:13px;border-radius:6px;background-color:#f5f5f6;cursor:pointer}.new-pmd .c-img{position:relative;display:block;min-height:0;border:0;line-height:0;background:#f5f5f6;overflow:hidden}.new-pmd .c-img img{width:100%}.new-pmd .c-img1{width:32px}.new-pmd .c-img2{width:80px}.new-pmd .c-img3{width:128px}.new-pmd .c-img4{width:176px}.new-pmd .c-img6{width:272px}.new-pmd .c-img12{width:560px}.new-pmd .c-img-s,.new-pmd .c-img-l,.new-pmd .c-img-w,.new-pmd .c-img-x,.new-pmd .c-img-y,.new-pmd .c-img-v,.new-pmd .c-img-z{height:0;overflow:hidden}.new-pmd .c-img-s{padding-bottom:100%}.new-pmd .c-img-l{padding-bottom:133.33333333%}.new-pmd .c-img-w{padding-bottom:56.25%}.new-pmd .c-img-x{padding-bottom:75%}.new-pmd .c-img-y{padding-bottom:66.66666667%}.new-pmd .c-img-v{padding-bottom:116.66666667%}.new-pmd .c-img-z{padding-bottom:62.5%}.new-pmd .c-img-radius{border-radius:6px}.new-pmd .c-img-radius-s{border-radius:2px}.new-pmd .c-img-radius-small{border-radius:2px}.new-pmd .c-img-radius-large{border-radius:12px}.new-pmd .c-img-radius-middle{border-radius:4px}.new-pmd .c-img-radius-left{border-top-left-radius:6px;border-bottom-left-radius:6px}.new-pmd .c-img-radius-right{border-top-right-radius:6px;border-bottom-right-radius:6px}.new-pmd .c-img-radius-left-s{border-top-left-radius:2px;border-bottom-left-radius:2px}.new-pmd .c-img-radius-right-s{border-top-right-radius:2px;border-bottom-right-radius:2px}.new-pmd .c-img-radius-left-l{border-top-left-radius:12px;border-bottom-left-radius:12px}.new-pmd .c-img-radius-right-l{border-top-right-radius:12px;border-bottom-right-radius:12px}.new-pmd .c-img-mask{position:absolute;top:0;left:0;z-index:2;width:100%;height:100%;background-image:radial-gradient(circle,rgba(0,0,0,0),rgba(0,0,0,.04));background-image:-ms-radial-gradient(circle,rgba(0,0,0,0),rgba(0,0,0,.04))}.new-pmd .c-img-border{content:&#39;&#39;;position:absolute;top:0;left:0;bottom:0;right:0;border:1px solid rgba(0,0,0,.05)}.new-pmd .c-img-circle{border-radius:100%;overflow:hidden}.new-pmd .c-input{display:inline-block;font:13px/21px Arial,sans-serif;color:#333;border:1px solid #D7D9E0;padding:0 8px;height:28px;line-height:28px\9;border-radius:6px;font-size:13px;outline:0;box-sizing:content-box;-webkit-box-sizing:content-box;-moz-box-sizing:content-box;vertical-align:top;overflow:hidden}.new-pmd .c-input:hover{box-shadow:none;-webkit-box-shadow:none;-moz-box-shadow:none;-o-box-shadow:none}.new-pmd .c-input .c-icon{float:right;margin-top:5px;font-size:16px;color:#9195A3}.new-pmd .c-input .c-icon-left{float:left;margin-right:4px}.new-pmd .c-input input{float:left;height:26px;padding:0;margin-top:1px;font-size:13px;border:0;outline:0}.new-pmd .c-input input::-webkit-input-placeholder{color:#9195A3}.new-pmd .c-input input::-ms-input-placeholder{color:#9195A3}.new-pmd .c-input input::-moz-placeholder{color:#9195A3}.new-pmd .c-input::-webkit-input-placeholder{color:#9195A3}.new-pmd .c-input::-ms-input-placeholder{color:#9195A3}.new-pmd .c-input::-moz-placeholder{color:#9195A3}.new-pmd .c-input{width:398px}.new-pmd .c-input input{width:378px}.new-pmd .c-input-xmini{width:158px}.new-pmd .c-input-xmini input{width:138px}.new-pmd .c-input-mini{width:206px}.new-pmd .c-input-mini input{width:186px}.new-pmd .c-input-small{width:350px}.new-pmd .c-input-small input{width:330px}.new-pmd .c-input-large{width:446px}.new-pmd .c-input-large input{width:426px}.new-pmd .c-input-xlarge{width:734px}.new-pmd .c-input-xlarge input{width:714px}.new-pmd .c-input12{width:542px}.new-pmd .c-input12 input{width:522px}.new-pmd .c-input20{width:926px}.new-pmd .c-input20 input{width:906px}.new-pmd .c-radio,.new-pmd .c-checkbox{display:inline-block;position:relative;white-space:nowrap;outline:0;line-height:1;vertical-align:middle;cursor:pointer;width:16px;height:16px}.new-pmd .c-radio-inner,.new-pmd .c-checkbox-inner{display:inline-block;position:relative;width:16px;height:16px;line-height:16px;text-align:center;top:0;left:0;background-color:#fff;color:#D7D9E0}.new-pmd .c-radio-input,.new-pmd .c-checkbox-input{position:absolute;top:0;bottom:0;left:0;right:0;z-index:1;opacity:0;filter:alpha(opacity=0) \9;user-select:none;margin:0;padding:0;width:100%;height:100%;cursor:pointer;zoom:1}.new-pmd .c-radio-inner-i,.new-pmd .c-checkbox-inner-i{display:none;font-size:16px}.new-pmd .c-radio-inner-bg,.new-pmd .c-checkbox-inner-bg{font-size:16px;position:absolute;top:0;left:0;z-index:1}.new-pmd .c-radio-checked .c-radio-inner-i,.new-pmd .c-checkbox-checked .c-checkbox-inner-i{color:#4E71F2;display:inline-block}.new-pmd .c-textarea{font:13px/21px Arial,sans-serif;color:#333;border:1px solid #D7D9E0;padding:8px 12px;border-radius:12px;resize:none;outline:0}.new-pmd .c-textarea::-webkit-input-placeholder{color:#9195A3}.new-pmd .c-textarea::-ms-input-placeholder{color:#9195A3}.new-pmd .c-textarea::-moz-placeholder{color:#9195A3}.new-pmd .c-table{width:100%;border-spacing:0;border-collapse:collapse}.new-pmd .c-table th,.new-pmd .c-table td{padding-left:10px;border-bottom:1px solid #f3f3f3;text-align:left;font-size:13px;line-height:1.54}.new-pmd .cr-content .c-table th:first-child,.new-pmd .cr-content .c-table td:first-child{padding-left:0}.new-pmd .c-table th{padding-top:4px;padding-bottom:4px;border-color:#f0f0f0;font-weight:400;white-space:nowrap;color:#666;background-color:#fafafa}.new-pmd .c-table td{padding-top:6.5px;padding-bottom:6.5px}.new-pmd .c-table-hasimg td{padding-top:10px;padding-bottom:10px}.new-pmd .c-table a,.new-pmd .c-table em{text-decoration:none}.new-pmd .c-table a:hover,.new-pmd .c-table a:hover em{text-decoration:underline}.new-pmd .c-table a.c-icon:hover{text-decoration:none}.new-pmd .c-table .c-btn:hover,.new-pmd .c-table .c-btn:hover em{text-decoration:none}.new-pmd .c-table-nohihead th{background-color:transparent}.new-pmd .c-table-noborder td{border-bottom:0}.new-pmd .c-tabs{font-size:14px;border-radius:12px;color:#222}.new-pmd .c-tabs-nav{color:#626675;background:#f5f5f6;border-radius:12px 12px 0 0;list-style:none;height:52px;margin:0;padding:0 12px}.new-pmd .c-tabs-nav-li{position:relative;display:inline-block;list-style:none;line-height:40px;height:40px;margin-right:32px;cursor:pointer}.new-pmd .c-tabs-nav-li:last-child{margin-right:0}.new-pmd .c-tabs-nav-selected{color:#222}.new-pmd .c-tabs-nav-selected::after{content:&#39;&#39;;position:absolute;bottom:0;height:2px;border-radius:1px;width:100%;left:0;z-index:1;background:#222}.new-pmd .c-tabs-content{padding:14px 16px;background:#fff;border-radius:12px;margin-top:-12px;box-shadow:0 2px 3px 0 rgba(0,0,0,.1);-webkit-box-shadow:0 2px 3px 0 rgba(0,0,0,.1);-moz-box-shadow:0 2px 3px 0 rgba(0,0,0,.1);-o-box-shadow:0 2px 3px 0 rgba(0,0,0,.1)}.new-pmd .c-tabs-nav-icon{display:inline-block;width:18px;height:18px;line-height:18px;border-radius:4px;margin-right:8px;background-size:contain;margin-top:11px;vertical-align:top}.new-pmd .c-tabs-nav-icon img{width:18px;height:18px}.new-pmd .c-tabs.c-sub-tabs .c-tabs-nav{height:29px;line-height:29px;border-bottom:1px solid #f2f2f2;background:#fff}.new-pmd .c-tabs.c-sub-tabs .c-tabs-content{box-shadow:none;-webkit-box-shadow:none;-moz-box-shadow:none;-o-box-shadow:none;margin-top:0;border-radius:0}.new-pmd .c-tabs.c-sub-tabs .c-tabs-nav-li{height:29px;line-height:29px}.new-pmd .c-tabs.c-sub-tabs .c-tabs-nav-icon{position:relative;margin-top:5px}.new-pmd .c-tabs.c-sub-tabs .c-tabs-nav-icon::after{content:&#39;&#39;;position:absolute;top:0;left:0;bottom:0;right:0;border:1px solid rgba(0,0,0,.03);border-radius:4px}.new-pmd .c-line-clamp1{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.new-pmd .c-line-clamp2{display:-webkit-box;overflow:hidden;-webkit-line-clamp:2;-webkit-box-orient:vertical}.new-pmd .c-font-sigma{font:36px/60px Arial,sans-serif}.new-pmd .c-font-large{font:18px/22px Arial,sans-serif}.new-pmd .c-font-big{font:18px/22px Arial,sans-serif}.new-pmd .c-font-special{font:16px/26px Arial,sans-serif}.new-pmd .c-font-medium{font:14px/22px Arial,sans-serif}.new-pmd .c-font-middle{font:14px/22px Arial,sans-serif}.new-pmd .c-font-normal{font:13px/21px Arial,sans-serif}.new-pmd .c-font-small{font:12px/20px Arial,sans-serif}.new-pmd .c-font-family{font-family:Arial,sans-serif}.new-pmd .c-color-t{color:#222}.new-pmd .c-color-text{color:#333}.new-pmd .c-color-gray{color:#626675}.new-pmd .c-color-gray2{color:#9195A3}.new-pmd .c-color-visited{color:#771CAA}.new-pmd .c-color-orange{color:#f60}.new-pmd .c-color-green{color:#00B198}.new-pmd .c-color-ad{color:#77A9F9}.new-pmd .c-color-red{color:#F73131}.new-pmd .c-color-red:visited{color:#F73131}.new-pmd .c-color-warn{color:#FF7900}.new-pmd .c-color-warn:visited{color:#FF7900}.new-pmd .c-color-link{color:#2440B3}.new-pmd .c-select{position:relative;display:inline-block;width:96px;box-sizing:border-box;-webkit-box-sizing:border-box;-moz-box-sizing:border-box;vertical-align:middle;color:#222;font:13px/21px Arial,sans-serif}.new-pmd .c-select-selection{display:block;height:30px;line-height:29px;box-sizing:border-box;-webkit-box-sizing:border-box;-moz-box-sizing:border-box;padding:0 26px 0 10px;background-color:#fff;border-radius:6px;border:1px solid #D7D9E0;outline:0;user-select:none;cursor:pointer;position:relative;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.new-pmd .c-select-arrow,.new-pmd .c-select-arrow-up{position:absolute;top:-1px;right:10px;color:#9195A3;font-size:16px}.new-pmd .c-select-dropdown{display:none;position:absolute;padding-top:4px;top:25px;z-index:999;left:0;width:94px;box-sizing:content-box;-webkit-box-sizing:content-box;-moz-box-sizing:content-box;background:#fff;border-radius:0 0 6px 6px;border:1px solid #D7D9E0;border-top:0;zoom:1}.new-pmd .c-select-split{border-top:1px solid #f5f5f5;margin:0 5px}.new-pmd .c-select-dropdown-list{padding:0;margin:5px 0 0;list-style:none}.new-pmd .c-select-dropdown-list.c-select-scroll{max-height:207px;overflow-y:auto;overflow-x:hidden;margin-right:5px;margin-bottom:9px}.new-pmd .c-select-dropdown-list.c-select-scroll::-webkit-scrollbar{width:2px}.new-pmd .c-select-dropdown-list.c-select-scroll::-webkit-scrollbar-track{width:2px;background:#f5f5f6;border-radius:1px}.new-pmd .c-select-dropdown-list.c-select-scroll::-webkit-scrollbar-thumb{width:2px;height:58px;background-color:#4e71f2;border-radius:1px}.new-pmd .c-select-dropdown-list.c-select-scroll .c-select-item:last-child{margin:0}.new-pmd .c-select-item{margin:0 0 4px;padding:0 10px;clear:both;white-space:nowrap;list-style:none;cursor:pointer;box-sizing:border-box;-webkit-box-sizing:border-box;-moz-box-sizing:border-box}.new-pmd .c-select-item:hover{color:#315EFB}.new-pmd .c-select-item-selected{color:#315EFB}.new-pmd .c-select-arrow-up{display:none}.new-pmd .c-select-visible .c-select-selection{border-radius:6px 6px 0 0}.new-pmd .c-select-visible .c-select-dropdown{display:block}.new-pmd .c-select-visible .c-select-arrow{display:none}.new-pmd .c-select-visible .c-select-arrow-up{display:inline-block}.new-pmd .c-frame{margin-bottom:18px}.new-pmd .c-offset{padding-left:10px}.new-pmd .c-link{color:#2440B3;text-decoration:none;cursor:pointer}.new-pmd .c-link:hover{text-decoration:underline;color:#315EFB}.new-pmd .c-link:visited{color:#771CAA}.new-pmd .c-gray{color:#626675}.new-pmd.c-container{width:560px;word-wrap:break-word;word-break:break-all;color:#333;font-size:13px;line-height:21px}.new-pmd.c-container .c-container{width:auto;font-size:13px;line-height:21px}.new-pmd .c-title{font:18px/22px Arial,sans-serif;font-weight:400;margin-bottom:4px}.new-pmd .c-abstract{font:13px/21px Arial,sans-serif;color:#222}.new-pmd .cr-title{font:14px/22px Arial,sans-serif;color:#222;font-weight:400}.new-pmd .cr-title-sub{float:right;font-weight:400;font-size:13px}.new-pmd .c-vline{display:inline-block;width:0;height:12px;margin:0 3px;border-left:1px solid #ddd}.new-pmd .c-border{border-radius:12px;border:0;margin:0 -16px;padding:12px 16px;width:auto;box-shadow:0 2px 5px 0 rgba(0,0,0,.1);-webkit-box-shadow:0 2px 5px 0 rgba(0,0,0,.1);-moz-box-shadow:0 2px 5px 0 rgba(0,0,0,.1);-o-box-shadow:0 2px 5px 0 rgba(0,0,0,.1)}.new-pmd .c-capsule-tip{display:inline-block;background:#F73131;border-radius:7px;padding:0 4px;height:13px;font-size:11px;line-height:14px;color:#fff;text-align:center}.c-group-wrapper{box-shadow:0 2px 10px 0 rgba(0,0,0,.1);border-radius:12px;margin-left:-16px;margin-right:-16px}.c-group-wrapper .result-op{padding:0 16px 16px;width:560px!important;border:0}.c-group-wrapper .result-op[id=&quot;1&quot;]{padding-top:16px}.c-group-wrapper .result-op:not(:last-child){margin-bottom:0!important}.c-group-wrapper .result-op:last-child{padding-bottom:16px}.c-group-title{font-size:14px;line-height:24px;font-weight:400;margin-bottom:4px}.c-group-title a{text-decoration:none;color:#222;line-height:24px}.c-group-title a:hover{color:#315EFB;text-decoration:none}.c-group-title a:hover&gt;i,.c-group-title a:hover+i,.c-group-title a:hover .c-group-arrow-icon{color:#315EFB!important}.c-group-title .c-group-arrow-icon{font-size:13px;line-height:13px;color:#c4c7ce;margin-left:-4px}#container.sam_newgrid{font:13px/21px Arial,sans-serif}#container.sam_newgrid td,#container.sam_newgrid th{font:13px/21px Arial,sans-serif}#container.sam_newgrid #content_left{width:560px}.container_l.sam_newgrid{width:1088px}.container_l.sam_newgrid #content_right{width:368px}.container_l.sam_newgrid .cr-content{width:368px}.container_l.sam_newgrid .cr-content .c-span-last-s{margin-right:16px}.container_l.sam_newgrid .cr-content-narrow .c-span-last-s{margin-right:0}.container_s.sam_newgrid{width:944px}.container_s.sam_newgrid .cr-content{width:272px}.container_s.sam_newgrid #content_right{width:272px}.c-onlyshow-toppic{width:100%;margin-top:-97px;padding-top:97px}.darkmode .new-pmd.c-container{color:#A8ACAD}.darkmode .new-pmd .c-abstract{color:#A8ACAD}.darkmode .new-pmd .c-link{color:#FFD862}.darkmode .new-pmd .c-link:hover{color:#FFF762}.darkmode .new-pmd .c-link:visited{color:#E7BDFF}.darkmode .new-pmd .c-btn{background-color:#31313B}.darkmode .new-pmd .c-btn,.darkmode .new-pmd .c-btn:visited{color:#A8ACAD!important}.darkmode .new-pmd .c-btn-disable{color:#6F7273!important}.darkmode .new-pmd .c-btn-disable:visited{color:#6F7273!important}.darkmode .new-pmd .c-btn-disable:hover{color:#6F7273!important;background-color:#31313B}.darkmode .new-pmd .c-btn-primary{color:#fff!important;background:#4E6EF2!important}.darkmode .new-pmd .c-btn-primary:visited{color:#fff!important;background:#4E6EF2!important}.darkmode .new-pmd .c-btn-add-disable:hover{color:#6F7273!important;background-color:#31313B}.darkmode .new-pmd .c-color-link{color:#FFD862}.darkmode .new-pmd .c-color-visited{color:#E7BDFF}.darkmode .new-pmd .c-color-t{color:#A8ACAD}.darkmode .new-pmd .c-color-text{color:#A8ACAD}.darkmode .new-pmd .c-color-red{color:#F14D2D}.darkmode .new-pmd .c-color-red:visited{color:#F14D2D}.darkmode .new-pmd .c-gray{color:#A8ACAD}.darkmode .new-pmd .c-color-gray{color:#A8ACAD!important}.darkmode .new-pmd .c-color-gray2{color:#A8ACAD!important}.darkmode .new-pmd .c-text-danger{background-color:#F14D2D}.darkmode .new-pmd .c-text-red{background-color:#F14D2D}.darkmode .new-pmd .c-text-red-border{color:#F14D2D!important}.darkmode .new-pmd .c-text-public{background-color:#6783F4}.darkmode .new-pmd .c-text-blue{background-color:#6783F4}.darkmode .new-pmd .c-text-blue-border{color:#6783F4!important}.darkmode .new-pmd .c-text-gray{background-color:#A5ABAC}.darkmode .new-pmd .c-text-gray-border{color:#A5ABAC!important}.darkmode .new-pmd .c-text-dark-red{background-color:#F74A4A}.darkmode .new-pmd .c-text-bao{background-color:#FF2D8B}.darkmode .new-pmd .c-capsule-tip{background:#F14D2D}.darkmode .new-pmd .c-select{color:#A8ACAD}.darkmode .new-pmd .c-select-arrow-up{color:#A8ACAD}.darkmode .new-pmd .c-select-item:hover{color:#FFF762}.darkmode .new-pmd .c-select-item-selected{color:#FFF762}.darkmode .new-pmd .c-tabs-nav{color:#A8ACAD}.c-pc-toppic-card{min-width:1116px}.pc-fresh-smooth .c-group-wrapper{position:relative;box-shadow:none;-webkit-box-shadow:none;-moz-box-shadow:none;-o-box-shadow:none;border-radius:var(--cos-space-md);margin:0 -15px}.pc-fresh-smooth .c-group-wrapper::after{content:&#39;&#39;;position:absolute;top:0;left:0;width:200%;height:200%;border:1px solid var(--cos-color-border-tiny);transform:scale(0.5);border-radius:calc(2 * var(--cos-rounded-lg));transform-origin:0 0;pointer-events:none;box-sizing:border-box}.pc-fresh-smooth .c-group-wrapper .result-op{padding:0 var(--cos-space-md) 16px}.pc-fresh-smooth .c-group-wrapper .result-op[id=&quot;1&quot;]{padding-top:var(--cos-space-lg)}.pc-fresh-smooth .c-group-wrapper .result-op:last-child{padding-bottom:var(--cos-space-md);margin-bottom:0!important}.pc-fresh-smooth #content_left .search-source-wrap{margin-top:var(--cos-space-xs);margin-bottom:20px}.pc-fresh-smooth .new-pmd .c-border{position:relative;box-shadow:none;-webkit-box-shadow:none;-moz-box-shadow:none;-o-box-shadow:none;border-radius:var(--cos-space-md);margin:0 -15px;padding:var(--cos-space-lg) var(--cos-space-md) var(--cos-space-md);width:auto}.pc-fresh-smooth .new-pmd .c-border::after{content:&#39;&#39;;position:absolute;top:0;left:0;width:200%;height:200%;border:1px solid var(--cos-color-border-tiny);transform:scale(0.5);border-radius:calc(2 * var(--cos-rounded-lg));transform-origin:0 0;pointer-events:none;box-sizing:border-box}.c-group-wrapper{box-shadow:0 2px 10px 0 rgba(0,0,0,.1);border-radius:12px;margin-left:-16px;margin-right:-16px}.c-group-wrapper .result-op{padding:0 16px 16px;width:560px!important;border:0}.c-group-wrapper .result-op[id=&quot;1&quot;]{padding-top:16px}.c-group-wrapper .result-op:not(:last-child){margin-bottom:0!important}.c-group-wrapper .result-op:last-child{padding-bottom:16px}.c-group-title{font-size:14px;line-height:24px;font-weight:400;margin-bottom:4px}.c-group-title a{text-decoration:none;color:#222;line-height:24px}.c-group-title a:hover{color:#315EFB;text-decoration:none}.c-group-title a:hover&gt;i,.c-group-title a:hover+i,.c-group-title a:hover .c-group-arrow-icon{color:#315EFB!important}.c-group-title .c-group-arrow-icon{font-size:13px;line-height:13px;color:#c4c7ce;margin-left:-4px}.pc-fresh-wrapper-ext{}.pc-fresh-wrapper-ext .new-pmd{}.pc-fresh-wrapper-ext .new-pmd .c-row{*zoom:1}.pc-fresh-wrapper-ext .new-pmd .c-row:after{display:block;height:0;content:&quot;&quot;;clear:both;visibility:hidden}.pc-fresh-wrapper-ext .new-pmd .c-span1{width:44px}.pc-fresh-wrapper-ext .new-pmd .c-span2{width:100px}.pc-fresh-wrapper-ext .new-pmd .c-span3{width:156px}.pc-fresh-wrapper-ext .new-pmd .c-span4{width:212px}.pc-fresh-wrapper-ext .new-pmd .c-span5{width:268px}.pc-fresh-wrapper-ext .new-pmd .c-span6{width:324px}.pc-fresh-wrapper-ext .new-pmd .c-span7{width:380px}.pc-fresh-wrapper-ext .new-pmd .c-span8{width:436px}.pc-fresh-wrapper-ext .new-pmd .c-span9{width:492px}.pc-fresh-wrapper-ext .new-pmd .c-span10{width:548px}.pc-fresh-wrapper-ext .new-pmd .c-span11{width:604px}.pc-fresh-wrapper-ext .new-pmd .c-span12{width:660px}.pc-fresh-wrapper-ext .new-pmd .c-span2,.pc-fresh-wrapper-ext .new-pmd .c-span3,.pc-fresh-wrapper-ext .new-pmd .c-span4,.pc-fresh-wrapper-ext .new-pmd .c-span5,.pc-fresh-wrapper-ext .new-pmd .c-span6,.pc-fresh-wrapper-ext .new-pmd .c-span7,.pc-fresh-wrapper-ext .new-pmd .c-span8,.pc-fresh-wrapper-ext .new-pmd .c-span9,.pc-fresh-wrapper-ext .new-pmd .c-span10,.pc-fresh-wrapper-ext .new-pmd .c-span11,.pc-fresh-wrapper-ext .new-pmd .c-span12{float:left;_display:inline;margin-right:12px;list-style:none}.pc-fresh-wrapper-ext .new-pmd .c-span-last{margin-right:0}.pc-fresh-wrapper-ext .new-pmd .c-span-last-s{margin-right:0}.pc-fresh-wrapper-ext .new-pmd .c-img{position:relative;display:block;min-height:0;border:0;line-height:0;background:#f5f5f6;overflow:hidden}.pc-fresh-wrapper-ext .new-pmd .c-img img{width:100%}.pc-fresh-wrapper-ext .new-pmd .c-img1{width:44px}.pc-fresh-wrapper-ext .new-pmd .c-img2{width:100px}.pc-fresh-wrapper-ext .new-pmd .c-img3{width:156px}.pc-fresh-wrapper-ext .new-pmd .c-img4{width:212px}.pc-fresh-wrapper-ext .new-pmd .c-img6{width:324px}.pc-fresh-wrapper-ext .new-pmd .c-img12{width:660px}.pc-fresh-wrapper-ext .new-pmd .c-img-s,.pc-fresh-wrapper-ext .new-pmd .c-img-l,.pc-fresh-wrapper-ext .new-pmd .c-img-w,.pc-fresh-wrapper-ext .new-pmd .c-img-x,.pc-fresh-wrapper-ext .new-pmd .c-img-y,.pc-fresh-wrapper-ext .new-pmd .c-img-v,.pc-fresh-wrapper-ext .new-pmd .c-img-z{height:0;overflow:hidden}.pc-fresh-wrapper-ext .new-pmd .c-img-s{padding-bottom:100%}.pc-fresh-wrapper-ext .new-pmd .c-img-l{padding-bottom:133.33333333%}.pc-fresh-wrapper-ext .new-pmd .c-img-w{padding-bottom:56.25%}.pc-fresh-wrapper-ext .new-pmd .c-img-x{padding-bottom:75%}.pc-fresh-wrapper-ext .new-pmd .c-img-y{padding-bottom:66.66666667%}.pc-fresh-wrapper-ext .new-pmd .c-img-v{padding-bottom:116.66666667%}.pc-fresh-wrapper-ext .new-pmd .c-img-z{padding-bottom:62.5%}.pc-fresh-wrapper-ext .new-pmd .c-img-radius{border-radius:6px}.pc-fresh-wrapper-ext .new-pmd .c-img-radius-s{border-radius:2px}.pc-fresh-wrapper-ext .new-pmd .c-img-radius-small{border-radius:2px}.pc-fresh-wrapper-ext .new-pmd .c-img-radius-large{border-radius:12px}.pc-fresh-wrapper-ext .new-pmd .c-img-radius-middle{border-radius:4px}.pc-fresh-wrapper-ext .new-pmd .c-img-radius-left{border-top-left-radius:6px;border-bottom-left-radius:6px}.pc-fresh-wrapper-ext .new-pmd .c-img-radius-right{border-top-right-radius:6px;border-bottom-right-radius:6px}.pc-fresh-wrapper-ext .new-pmd .c-img-radius-left-s{border-top-left-radius:2px;border-bottom-left-radius:2px}.pc-fresh-wrapper-ext .new-pmd .c-img-radius-right-s{border-top-right-radius:2px;border-bottom-right-radius:2px}.pc-fresh-wrapper-ext .new-pmd .c-img-radius-left-l{border-top-left-radius:12px;border-bottom-left-radius:12px}.pc-fresh-wrapper-ext .new-pmd .c-img-radius-right-l{border-top-right-radius:12px;border-bottom-right-radius:12px}.pc-fresh-wrapper-ext .new-pmd .c-img-mask{position:absolute;top:0;left:0;z-index:2;width:100%;height:100%;background-image:radial-gradient(circle,rgba(0,0,0,0),rgba(0,0,0,.04));background-image:-ms-radial-gradient(circle,rgba(0,0,0,0),rgba(0,0,0,.04))}.pc-fresh-wrapper-ext .new-pmd .c-img-border{content:&#39;&#39;;position:absolute;top:0;left:0;bottom:0;right:0;border:1px solid rgba(0,0,0,.05)}.pc-fresh-wrapper-ext .new-pmd .c-img-circle{border-radius:100%;overflow:hidden}.pc-fresh-wrapper-ext .new-pmd.c-container{width:660px}.pc-fresh-wrapper-ext #container.sam_newgrid #content_left{width:660px}.pc-fresh-wrapper-ext .container_l.sam_newgrid,.pc-fresh-wrapper-ext .container_s.sam_newgrid{width:1108px}.pc-fresh-wrapper-ext .container_l.sam_newgrid #content_right,.pc-fresh-wrapper-ext .container_s.sam_newgrid #content_right{width:324px}.pc-fresh-wrapper-ext .container_l.sam_newgrid .cr-content,.pc-fresh-wrapper-ext .container_s.sam_newgrid .cr-content{width:324px}.pc-fresh-wrapper-ext .container_l.sam_newgrid .cr-content .c-span-last-s,.pc-fresh-wrapper-ext .container_s.sam_newgrid .cr-content .c-span-last-s{margin-right:12px}.pc-fresh-wrapper-ext .c-group-wrapper .result-op{width:660px!important}.pc-fresh-wrapper-ext .c-pc-toppic-card{min-width:1412px}.c-group-wrapper{box-shadow:0 2px 10px 0 rgba(0,0,0,.1);border-radius:12px;margin-left:-16px;margin-right:-16px}.c-group-wrapper .result-op{padding:0 16px 16px;width:560px!important;border:0}.c-group-wrapper .result-op[id=&quot;1&quot;]{padding-top:16px}.c-group-wrapper .result-op:not(:last-child){margin-bottom:0!important}.c-group-wrapper .result-op:last-child{padding-bottom:16px}.c-group-title{font-size:14px;line-height:24px;font-weight:400;margin-bottom:4px}.c-group-title a{text-decoration:none;color:#222;line-height:24px}.c-group-title a:hover{color:#315EFB;text-decoration:none}.c-group-title a:hover&gt;i,.c-group-title a:hover+i,.c-group-title a:hover .c-group-arrow-icon{color:#315EFB!important}.c-group-title .c-group-arrow-icon{font-size:13px;line-height:13px;color:#c4c7ce;margin-left:-4px}.pc-fresh-wrapper-con{}.pc-fresh-wrapper-con .new-pmd{}.pc-fresh-wrapper-con .new-pmd .c-row{*zoom:1}.pc-fresh-wrapper-con .new-pmd .c-row:after{display:block;height:0;content:&quot;&quot;;clear:both;visibility:hidden}.pc-fresh-wrapper-con .new-pmd .c-span1{width:36px}.pc-fresh-wrapper-con .new-pmd .c-span2{width:88px}.pc-fresh-wrapper-con .new-pmd .c-span3{width:140px}.pc-fresh-wrapper-con .new-pmd .c-span4{width:192px}.pc-fresh-wrapper-con .new-pmd .c-span5{width:244px}.pc-fresh-wrapper-con .new-pmd .c-span6{width:296px}.pc-fresh-wrapper-con .new-pmd .c-span7{width:348px}.pc-fresh-wrapper-con .new-pmd .c-span8{width:400px}.pc-fresh-wrapper-con .new-pmd .c-span9{width:452px}.pc-fresh-wrapper-con .new-pmd .c-span10{width:504px}.pc-fresh-wrapper-con .new-pmd .c-span11{width:556px}.pc-fresh-wrapper-con .new-pmd .c-span12{width:608px}.pc-fresh-wrapper-con .new-pmd .c-span2,.pc-fresh-wrapper-con .new-pmd .c-span3,.pc-fresh-wrapper-con .new-pmd .c-span4,.pc-fresh-wrapper-con .new-pmd .c-span5,.pc-fresh-wrapper-con .new-pmd .c-span6,.pc-fresh-wrapper-con .new-pmd .c-span7,.pc-fresh-wrapper-con .new-pmd .c-span8,.pc-fresh-wrapper-con .new-pmd .c-span9,.pc-fresh-wrapper-con .new-pmd .c-span10,.pc-fresh-wrapper-con .new-pmd .c-span11,.pc-fresh-wrapper-con .new-pmd .c-span12{float:left;_display:inline;margin-right:16px;list-style:none}.pc-fresh-wrapper-con .new-pmd .c-span-last{margin-right:0}.pc-fresh-wrapper-con .new-pmd .c-span-last-s{margin-right:0}.pc-fresh-wrapper-con .new-pmd .c-img{position:relative;display:block;min-height:0;border:0;line-height:0;background:#f5f5f6;overflow:hidden}.pc-fresh-wrapper-con .new-pmd .c-img img{width:100%}.pc-fresh-wrapper-con .new-pmd .c-img1{width:36px}.pc-fresh-wrapper-con .new-pmd .c-img2{width:88px}.pc-fresh-wrapper-con .new-pmd .c-img3{width:140px}.pc-fresh-wrapper-con .new-pmd .c-img4{width:192px}.pc-fresh-wrapper-con .new-pmd .c-img6{width:296px}.pc-fresh-wrapper-con .new-pmd .c-img12{width:608px}.pc-fresh-wrapper-con .new-pmd .c-img-s,.pc-fresh-wrapper-con .new-pmd .c-img-l,.pc-fresh-wrapper-con .new-pmd .c-img-w,.pc-fresh-wrapper-con .new-pmd .c-img-x,.pc-fresh-wrapper-con .new-pmd .c-img-y,.pc-fresh-wrapper-con .new-pmd .c-img-v,.pc-fresh-wrapper-con .new-pmd .c-img-z{height:0;overflow:hidden}.pc-fresh-wrapper-con .new-pmd .c-img-s{padding-bottom:100%}.pc-fresh-wrapper-con .new-pmd .c-img-l{padding-bottom:133.33333333%}.pc-fresh-wrapper-con .new-pmd .c-img-w{padding-bottom:56.25%}.pc-fresh-wrapper-con .new-pmd .c-img-x{padding-bottom:75%}.pc-fresh-wrapper-con .new-pmd .c-img-y{padding-bottom:66.66666667%}.pc-fresh-wrapper-con .new-pmd .c-img-v{padding-bottom:116.66666667%}.pc-fresh-wrapper-con .new-pmd .c-img-z{padding-bottom:62.5%}.pc-fresh-wrapper-con .new-pmd .c-img-radius{border-radius:6px}.pc-fresh-wrapper-con .new-pmd .c-img-radius-s{border-radius:2px}.pc-fresh-wrapper-con .new-pmd .c-img-radius-small{border-radius:2px}.pc-fresh-wrapper-con .new-pmd .c-img-radius-large{border-radius:12px}.pc-fresh-wrapper-con .new-pmd .c-img-radius-middle{border-radius:4px}.pc-fresh-wrapper-con .new-pmd .c-img-radius-left{border-top-left-radius:6px;border-bottom-left-radius:6px}.pc-fresh-wrapper-con .new-pmd .c-img-radius-right{border-top-right-radius:6px;border-bottom-right-radius:6px}.pc-fresh-wrapper-con .new-pmd .c-img-radius-left-s{border-top-left-radius:2px;border-bottom-left-radius:2px}.pc-fresh-wrapper-con .new-pmd .c-img-radius-right-s{border-top-right-radius:2px;border-bottom-right-radius:2px}.pc-fresh-wrapper-con .new-pmd .c-img-radius-left-l{border-top-left-radius:12px;border-bottom-left-radius:12px}.pc-fresh-wrapper-con .new-pmd .c-img-radius-right-l{border-top-right-radius:12px;border-bottom-right-radius:12px}.pc-fresh-wrapper-con .new-pmd .c-img-mask{position:absolute;top:0;left:0;z-index:2;width:100%;height:100%;background-image:radial-gradient(circle,rgba(0,0,0,0),rgba(0,0,0,.04));background-image:-ms-radial-gradient(circle,rgba(0,0,0,0),rgba(0,0,0,.04))}.pc-fresh-wrapper-con .new-pmd .c-img-border{content:&#39;&#39;;position:absolute;top:0;left:0;bottom:0;right:0;border:1px solid rgba(0,0,0,.05)}.pc-fresh-wrapper-con .new-pmd .c-img-circle{border-radius:100%;overflow:hidden}.pc-fresh-wrapper-con .new-pmd.c-container{width:608px}.pc-fresh-wrapper-con #container.sam_newgrid #content_left{width:608px}.pc-fresh-wrapper-con .container_l.sam_newgrid,.pc-fresh-wrapper-con .container_s.sam_newgrid{width:1076px}.pc-fresh-wrapper-con .container_l.sam_newgrid #content_right,.pc-fresh-wrapper-con .container_s.sam_newgrid #content_right{width:348px}.pc-fresh-wrapper-con .container_l.sam_newgrid .cr-content,.pc-fresh-wrapper-con .container_s.sam_newgrid .cr-content{width:348px}.pc-fresh-wrapper-con .container_l.sam_newgrid .cr-content .c-span-last-s,.pc-fresh-wrapper-con .container_s.sam_newgrid .cr-content .c-span-last-s{margin-right:16px}.pc-fresh-wrapper-con .c-group-wrapper .result-op{width:608px!important}.pc-fresh-wrapper-con .c-pc-toppic-card{min-width:1380px}.soutu-input{padding-left:55px!important}.soutu-input-image{position:absolute;left:1px;top:1px;height:28px;width:49px;z-index:1;padding:0;background:#e6e6e6;border:1px solid #e6e6e6}.soutu-input-thumb{height:28px;width:28px;min-width:1px}.soutu-input-close{position:absolute;right:0;top:0;cursor:pointer;display:block;width:22px;height:28px}.soutu-input-close::after{content:&quot; &quot;;position:absolute;right:3px;top:50%;cursor:pointer;margin-top:-7px;display:block;width:14px;height:14px;background:url(https://pss.bdstatic.com/r/www/cache/static/protocol/https/soutu/img/soutu_icons_new_8abaf8a.png) no-repeat -163px 0}.soutu-input-image:hover .soutu-input-close::after{background-position:-215px 2px}.fb-hint{margin-top:5px;transition-duration:.9s;opacity:0;display:none;color:red}.fb-img{display:none}.fb-hint-tip{height:44px;line-height:24px;background-color:#38f;color:#fff;box-sizing:border-box;width:269px;font-size:16px;padding:10px;padding-left:14px;position:absolute;top:-65px;right:-15px;border-radius:3px;z-index:299}.fb-hint-tip::before{content:&quot;&quot;;width:0;height:0;display:block;position:absolute;border-left:8px solid transparent;border-right:8px solid transparent;border-top:8px solid #38f;bottom:-8px;right:25px}.fb-mask,.fb-mask-light{position:fixed;top:0;left:0;bottom:0;right:0;z-index:296;background-color:#000;filter:alpha(opacity=60);background-color:rgba(0,0,0,.6)}.fb-mask-light{background-color:#fff;filter:alpha(opacity=0);background-color:rgba(255,255,255,0)}.fb-success .fb-success-text{text-align:center;color:#333;font-size:13px;margin-bottom:14px}.fb-success-text.fb-success-text-title{color:#3b6;font-size:16px;margin-bottom:16px}.fb-success-text-title i{width:16px;height:16px;margin-right:5px}.fb-list-container{box-sizing:border-box;padding:4px 8px;position:absolute;top:0;left:0;bottom:0;right:0;z-index:298;display:block;width:100%;cursor:pointer;margin-top:-5px;margin-left:-5px}.fb-list-container-hover{background-color:#fff;border:2px #38f solid}.fb-list-container-first{box-sizing:border-box;padding-left:10px;padding-top:5px;position:absolute;top:0;left:0;bottom:0;right:0;z-index:297;display:block;width:100%;cursor:pointer;margin-top:-5px;margin-left:-5px;border:3px #f5f5f5 dashed;border-radius:3px}.fb-des-content{font-size:13px!important;color:#000}.fb-des-content::-webkit-input-placeholder{font-size:13px!important;color:#9a9a9a}.fb-des-content:-moz-placeholder{font-size:13px!important;color:#9a9a9a}.fb-des-content::-moz-placeholder{font-size:13px!important;color:#9a9a9a}.fb-des-content:-ms-input-placeholder{font-size:13px!important;color:#9a9a9a}.fb-btn,.fb-btn:visited{color:#333!important}.fb-select{position:relative;background-color:#fff;border:1px solid #ccc}.fb-select i{position:absolute;right:2px;top:7px}.fb-type{width:350px;box-sizing:border-box;height:28px;font-size:13px;line-height:28px;border:0;word-break:normal;word-wrap:normal;position:relative;appearance:none;-moz-appearance:none;-webkit-appearance:none;display:inline-block;vertical-align:middle;line-height:normal;color:#333;background-color:transparent;border-radius:0;overflow:hidden;outline:0;padding-left:5px}.fb-type::-ms-expand{display:none}.fb-btn{display:inline-block;padding:0 14px;margin:0;height:24px;line-height:25px;font-size:13px;filter:chroma(color=#000000);*zoom:1;border:1px solid #d8d8d8;cursor:pointer;font-family:inherit;font-weight:400;text-align:center;vertical-align:middle;background-color:#f9f9f9;overflow:hidden;outline:0}.fb-btn:hover{border-color:#388bff}.fb-btn:active{border-color:#a2a6ab;background-color:#f0f0f0;box-shadow:inset 1px 1px 1px #c7c7c7;-webkit-box-shadow:inset 1px 1px 1px #c7c7c7;-moz-box-shadow:inset 1px 1px 1px #c7c7c7;-o-box-shadow:inset 1px 1px 1px #c7c7c7}a.fb-btn{text-decoration:none}button.fb-btn{height:26px;_line-height:18px;*overflow:visible}button.fb-btn::-moz-focus-inner{padding:0;border:0}.fb-btn .c-icon{margin-top:5px}.fb-btn-primary,.fb-btn-primary:visited{color:#fff!important}.fb-btn-primary{background-color:#388bff;_width:82px;border-color:#3c8dff #408ffe #3680e6}.fb-btn-primary:hover{border-color:#2678ec #2575e7 #1c6fe2 #2677e7;background-color:#388bff;background-image:url(data:image/png;    base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAACCAMAAACuX0YVAAAABlBMVEVnpv85i/9PO5r4AAAAD0lEQVR42gEEAPv/AAAAAQAFAAIros7PAAAAAElFTkSuQmCC);background-repeat:repeat-x;box-shadow:1px 1px 1px rgba(0,0,0,.4);-webkit-box-shadow:1px 1px 1px rgba(0,0,0,.4);-moz-box-shadow:1px 1px 1px rgba(0,0,0,.4);-o-box-shadow:1px 1px 1px rgba(0,0,0,.4)}.fb-btn-primary:active{border-color:#178ee3 #1784d0 #177bbf #1780ca;background-color:#388bff;background-image:none;box-shadow:inset 1px 1px 1px rgba(0,0,0,.15);-webkit-box-shadow:inset 1px 1px 1px rgba(0,0,0,.15);-moz-box-shadow:inset 1px 1px 1px rgba(0,0,0,.15);-o-box-shadow:inset 1px 1px 1px rgba(0,0,0,.15)}.fb-feedback-right-dialog{position:fixed;z-index:299;bottom:0;right:0}.fb-feedback-list-dialog,.fb-feedback-list-dialog-left{position:absolute;z-index:299}.fb-feedback-list-dialog:before{content:&quot;&quot;;width:0;height:0;display:block;position:absolute;top:15px;left:-6px;border-top:8px solid transparent;border-bottom:8px solid transparent;border-right:8px solid #fff}.fb-feedback-list-dialog-left:before{content:&quot;&quot;;width:0;height:0;display:block;position:absolute;top:15px;right:-6px;border-top:8px solid transparent;border-bottom:8px solid transparent;border-left:8px solid #fff}.fb-header{padding-left:20px;padding-right:20px;margin-top:14px;text-align:left;-moz-user-select:none}.fb-header .fb-close{color:#e0e0e0}.fb-close{text-decoration:none;margin-top:2px;float:right;font-size:20px;font-weight:700;line-height:18px;color:#666;text-shadow:0 1px 0 #fff}.fb-photo-block{display:none}.fb-photo-block-title{font-size:13px;color:#333;padding-top:10px}.fb-photo-block-title-span{color:#999}.fb-photo-sub-block{margin-top:10px;margin-bottom:10px;width:60px;text-align:center}.fb-photo-sub-block-hide{display:none}.fb-photo-update-block{overflow:hidden}.fb-photo-update-item-block{width:100px;height:100px;background:red;border:solid 1px #ccc;margin-top:10px;float:left;margin-right:20px;position:relative;background:url(https://pss.bdstatic.com/r/www/cache/static/protocol/https/global/img/feedback_add_photo_69ff822.png);background-repeat:no-repeat;background-size:contain;background-position:center center;background-size:24px 24px}.fb-photo-block-title-ex{font-size:13px;float:right}.fb-photo-block-title-ex img{vertical-align:text-top;margin-right:4px}.fb-photo-block-title-span{margin-left:4px;color:#999}.fb-photo-update-item-show-img{width:100%;height:100%;display:none}.fb-photo-update-item-close{width:13px;height:13px;position:absolute;top:-6px;right:-6px;display:none}.fb-photo-block input{display:none}.fb-photo-update-hide{display:none}.fb-photo-update-item-block{width:60px;height:60px;border:solid 1px #ccc;float:left}.fb-photo-block-example{position:absolute;top:0;left:0;display:none;background-color:#fff;padding:14px;padding-top:0;width:392px}.fb-photo-block-example-header{padding-top:14px;overflow:hidden}.fb-photo-block-example-header p{float:left}.fb-photo-block-example-header img{float:right;width:13px;height:13px}.fb-photo-block-example-img img{margin:0 auto;margin-top:14px;display:block;width:200px}.fb-photo-block-example-title{text-align:center}.fb-photo-block-example-title-big{font-size:14px;color:#333}.fb-photo-block-example-title-small{font-size:13px;color:#666}.fb-header a.fb-close:hover{text-decoration:none}.fb-photo-block-upinfo{width:100%}.fb-header-tips{font-size:16px;margin:0;color:#333;text-rendering:optimizelegibility}.fb-body{margin-bottom:0;padding:20px;padding-top:10px;overflow:hidden;text-align:left}.fb-modal,.fb-success,.fb-vertify{background-color:#fff;cursor:default;top:100%;left:100%;width:390px;overflow:hidden;border:1px solid #999;*border:1px solid #ddd;font-size:13px;line-height:1.54}.fb-textarea textarea{width:350px;height:64px;padding:4px;margin:10px 0;vertical-align:top;resize:none;overflow:auto;box-sizing:border-box;display:inline-block;border:1px solid #ccc;-webkit-border-radius:0;-moz-border-radius:0;border-radius:0;-webkit-box-shadow:inset 0 1px 1px rgba(0,0,0,.075);-moz-box-shadow:inset 0 1px 1px rgba(0,0,0,.075);box-shadow:inset 0 1px 1px rgba(0,0,0,.075);-webkit-transition:border linear .2s,box-shadow linear .2s;-moz-transition:border linear .2s,box-shadow linear .2s;-ms-transition:border linear .2s,box-shadow linear .2s;-o-transition:border linear .2s,box-shadow linear .2s;transition:border linear .2s,box-shadow linear .2s}.fb-selected{display:none;width:12px;height:12px;background:url(data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAcAAAAFCAYAAACJmvbYAAAABGdBTUEAALGPC/xhBQAAACBjSFJNAAB6JgAAgIQAAPoAAACA6AAAdTAAAOpgAAA6mAAAF3CculE8AAAABmJLR0QAAAAAAAD5Q7t/AAAACXBIWXMAABYlAAAWJQFJUiTwAAAAJklEQVQI12NgwAEsuv/8xy9h3vX7P6oEKp/BHCqA0yhzdB0MDAwAFXkTK5la4mAAAAAASUVORK5CYII=) no-repeat 2px 3px}.fb-guide{padding-top:10px;color:#9a9a9a;margin-left:-20px;padding-left:20px;border-right-width:0;margin-right:-20px;padding-right:25px;margin-bottom:-20px;padding-bottom:15px}.fb-footer{padding-top:10px;text-align:left}.fb-block{overflow:hidden;position:relative}.fb-block .fb-email{height:28px;line-height:26px;width:350px;border:1px solid #ccc;padding:4px;padding-top:0;box-sizing:border-box;padding-bottom:0;display:inline-block;font-family:&#39;Helvetica Neue&#39;,Helvetica,Arial,sans-serif;vertical-align:middle!important;-webkit-border-radius:0;-moz-border-radius:0;border-radius:0;-webkit-box-shadow:inset 0 1px 1px rgba(0,0,0,.075);-moz-box-shadow:inset 0 1px 1px rgba(0,0,0,.075);box-shadow:inset 0 1px 1px rgba(0,0,0,.075);-webkit-transition:border linear .2s,box-shadow linear .2s;-moz-transition:border linear .2s,box-shadow linear .2s;-ms-transition:border linear .2s,box-shadow linear .2s;-o-transition:border linear .2s,box-shadow linear .2s;transition:border linear .2s,box-shadow linear .2s}.fb-email{font-size:13px!important;color:#000}.fb-email::-webkit-input-placeholder{font-size:13px!important;color:#9a9a9a}.fb-email:-moz-placeholder{font-size:13px!important;color:#9a9a9a}.fb-email::-moz-placeholder{font-size:13px!important;color:#9a9a9a}.fb-email:-ms-input-placeholder{font-size:13px!important;color:#9a9a9a}.fb-cut-block{height:15px;padding-bottom:10px}.fb-canvas-block{height:172px;border:1px solid #ccc;margin-bottom:10px;position:relative;overflow:hidden;width:100%;background-position:center;box-sizing:border-box}.fb-canvas-block img{width:350px;position:absolute}.fb-canvas-block img[src=&quot;&quot;]{opacity:0}.fb-cut-input{width:14px;height:14px;margin:0;margin-right:10px;display:inline-block;border:1px solid #ccc}.fb-cut-btn{width:60px!important}#fb_tips_span{vertical-align:middle}#fb_popwindow{display:block;left:457px;top:69.5px;position:absolute;width:450px;z-index:999999;background:none repeat scroll 0 0 #fff;border:1px solid #999;border-radius:3px;box-shadow:0 0 9px #999;padding:0}#feedback_dialog_content{text-align:center}#fb_right_post_save:hover{background-image:url(data:image/png;    base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAACCAMAAACuX0YVAAAABlBMVEVnpv85i/9PO5r4AAAAD0lEQVR42gEEAPv/AAAAAQAFAAIros7PAAAAAElFTkSuQmCC);background-repeat:repeat-x;box-shadow:1px 1px 1px rgba(0,0,0,.4);-webkit-box-shadow:1px 1px 1px rgba(0,0,0,.4);-moz-box-shadow:1px 1px 1px rgba(0,0,0,.4);-o-box-shadow:1px 1px 1px rgba(0,0,0,.4)}.fb-select-icon{position:absolute;bottom:6px;right:5px;width:16px;height:16px;box-sizing:content-box;background-position:center center;background-repeat:no-repeat;background-size:7px 4px;-webkit-background-size:7px 4px;background-image:url(data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAcAAAAECAYAAABCxiV9AAAABGdBTUEAALGPC/xhBQAAACBjSFJNAAB6JgAAgIQAAPoAAACA6AAAdTAAAOpgAAA6mAAAF3CculE8AAAABmJLR0QAAAAAAAD5Q7t/AAAACXBIWXMAAAsSAAALEgHS3X78AAAAKElEQVQI12Ps7Or6z4ADMDIwMDBgU1BeVsbICOMgKygvK2PEMAbdBAAhxA08t5Q3VgAAAABJRU5ErkJggg==)}.fb-select-shorter{position:relative;min-height:28px}.fb-type-container{line-height:28px;position:absolute;top:28px;width:100%;background-color:#fff;border:1px solid #ccc;z-index:300;margin-left:-1px;display:none}.fb-type-item,.fb-type-selected{height:28px;line-height:30px;padding-left:4px}.fb-type-item:hover{background:#f5F5F5}.fb-checkbox{position:relative;border-bottom:1px solid #eee;height:34px;line-height:35px}.fb-checkbox:last-child{border-bottom:0}.fb-list-wrapper{margin-top:-10px;max-height:370px;overflow-y:scroll}.fb-list-wrapper::-webkit-scrollbar{background:#fff;width:10px;height:1px}.fb-list-wrapper::-webkit-scrollbar-thumb{background-color:#e0e0e0;border-radius:10px}.fb-textarea-sug textarea{margin-top:0}@media screen and (min-width:1921px){body:not(.pc-fresh-wrapper){}body:not(.pc-fresh-wrapper) .slowmsg{left:50%!important;-webkit-transform:translateX(-50%);-ms-transform:translateX(-50%);transform:translateX(-50%)}body:not(.pc-fresh-wrapper) .wrapper_l #head{-webkit-transform-style:preserve-3d;transform-style:preserve-3d}body:not(.pc-fresh-wrapper) .head_wrapper{width:1196px;margin:0 auto;position:relative;-webkit-transform:translate3d(-52px,0,1px);transform:translate3d(-52px,0,1px)}body:not(.pc-fresh-wrapper) .head_wrapper #u{right:-66px}body:not(.pc-fresh-wrapper) #head .headBlock{-webkit-box-sizing:border-box;box-sizing:border-box;margin-left:auto;margin-right:auto;width:1196px;padding-left:121px;-webkit-transform:translate3d(-52px,0,0);transform:translate3d(-52px,0,0)}body:not(.pc-fresh-wrapper) #s_tab.s_tab{padding-left:0}body:not(.pc-fresh-wrapper) #s_tab.s_tab .s_tab_inner{display:block;-webkit-box-sizing:border-box;box-sizing:border-box;padding-left:77px;width:1212px;margin:0 auto}body:not(.pc-fresh-wrapper) #s_tab.s_tab .s_tab_inner.s-new-tag{display:flex}body:not(.pc-fresh-wrapper) #s_tab.s_tab .s_tab_inner.s-new-tag .cur-tab{margin-right:4px}body:not(.pc-fresh-wrapper) #con-at .result-op{margin-left:auto;margin-right:auto;position:relative;left:-60px}body:not(.pc-fresh-wrapper) #wrapper_wrapper{margin-left:-72px}body:not(.pc-fresh-wrapper) #container{-webkit-box-sizing:border-box;box-sizing:border-box;width:1212px;margin:0 auto}body:not(.pc-fresh-wrapper) #container.sam_newgrid{margin:0 auto;width:1088px;padding-left:158px;-webkit-box-sizing:content-box;box-sizing:content-box}body:not(.pc-fresh-wrapper) .wrapper_l.new_input_wrapper #main-wrapper{transform:translateX(-2px)}body:not(.pc-fresh-wrapper) .wrapper_new #s_tab.s_tab .s_tab_inner{padding-left:105px}}@font-face{font-family:cicons;font-weight:400;font-style:normal;src:url(//m.baidu.com/se/static/font/cicon.eot?t=1743055862713#);src:url(//m.baidu.com/se/static/font/cicon.eot?t=1743055862713#iefix) format(&#39;embedded-opentype&#39;),url(//m.baidu.com/se/static/font/cicon.woff?t=1743055862713#) format(&#39;woff&#39;),url(//m.baidu.com/se/static/font/cicon.ttf?t=1743055862713#) format(&#39;truetype&#39;),url(//m.baidu.com/se/static/font/cicon.svg?t=1743055862713#cicons) format(&#39;svg&#39;)}@font-face{font-family:cIconfont;font-weight:400;font-style:normal;src:url(https://pss.bdstatic.com/r/www/cache/static/protocol/https/global/font/iconfont_1b4ac26.eot);src:url(https://pss.bdstatic.com/r/www/cache/static/protocol/https/global/font/iconfont_1b4ac26.eot?#iefix) format(&#39;embedded-opentype&#39;),url(https://pss.bdstatic.com/r/www/cache/static/protocol/https/global/font/iconfont_13bc846.woff2) format(&#39;woff2&#39;),url(https://pss.bdstatic.com/r/www/cache/static/protocol/https/global/font/iconfont_a4ef80b.woff) format(&#39;woff&#39;),url(https://pss.bdstatic.com/r/www/cache/static/protocol/https/global/font/iconfont_aebd236.ttf) format(&#39;truetype&#39;),url(https://pss.bdstatic.com/r/www/cache/static/protocol/https/global/font/iconfont_5d617bf.svg#iconfont) format(&#39;svg&#39;)}@font-face{font-family:cosmicIcon;font-weight:400;font-style:normal;src:url(https://pss.bdstatic.com/r/www/cache/static/protocol/https/global/font/cosmic-icon/iconfont_2a26dfd.eot);src:url(https://pss.bdstatic.com/r/www/cache/static/protocol/https/global/font/cosmic-icon/iconfont_2a26dfd.eot?#iefix) format(&#39;embedded-opentype&#39;),url(https://pss.bdstatic.com/r/www/cache/static/protocol/https/global/font/cosmic-icon/iconfont_728c86e.woff2) format(&#39;woff2&#39;),url(https://pss.bdstatic.com/r/www/cache/static/protocol/https/global/font/cosmic-icon/iconfont_04a7284.woff) format(&#39;woff&#39;),url(https://pss.bdstatic.com/r/www/cache/static/protocol/https/global/font/cosmic-icon/iconfont_96c2041.ttf) format(&#39;truetype&#39;),url(https://pss.bdstatic.com/r/www/cache/static/protocol/https/global/font/cosmic-icon/iconfont_dffe326.svg#iconfont) format(&#39;svg&#39;)}@font-face{font-family:DINPro;font-weight:400;font-style:normal;src:url(https://pss.bdstatic.com/r/www/cache/static/protocol/https/global/font/din-pro-cond-medium/DINPro-CondMedium_8b5bddc.eot);src:url(https://pss.bdstatic.com/r/www/cache/static/protocol/https/global/font/din-pro-cond-medium/DINPro-CondMedium_8b5bddc.eot) format(&#39;embedded-opentype&#39;),url(https://pss.bdstatic.com/r/www/cache/static/protocol/https/global/font/din-pro-cond-medium/DINPro-CondMedium_871bdc0.woff2) format(&#39;woff2&#39;),url(https://pss.bdstatic.com/r/www/cache/static/protocol/https/global/font/din-pro-cond-medium/DINPro-CondMedium_8ba4bba.woff) format(&#39;woff&#39;),url(https://pss.bdstatic.com/r/www/cache/static/protocol/https/global/font/din-pro-cond-medium/DINPro-CondMedium_419c346.ttf) format(&#39;truetype&#39;),url(https://pss.bdstatic.com/r/www/cache/static/protocol/https/global/font/din-pro-cond-medium/DINPro-CondMedium_7fcf171.svg#DINPro) format(&#39;svg&#39;)}@font-face{font-family:baidunumber-Medium;font-weight:400;font-style:normal;src:url(https://psstatic.cdn.bcebos.com/aladdin/baidunumber_Medium_1660569289000.woff) format(&quot;woff&quot;),url(https://psstatic.cdn.bcebos.com/aladdin/baidunumber_Medium_1660569280000.ttf) format(&quot;truetype&quot;),url(https://psstatic.cdn.bcebos.com/aladdin/baidunumber_Medium_1660569188000.svg) format(&quot;svg&quot;)}@font-face{font-family:LamaSansBlackItalic;src:url(https://pss.bdstatic.com/r/www/cache/static/protocol/https/global/font/lama-sans/LamaSans-BlackItalic_3ee8e8e.ttf) format(&#39;truetype&#39;),url(https://pss.bdstatic.com/r/www/cache/static/protocol/https/global/font/lama-sans/LamaSans-BlackItalic_648805b.woff) format(&#39;woff&#39;),url(https://pss.bdstatic.com/r/www/cache/static/protocol/https/global/font/lama-sans/LamaSans-BlackItalic_3c3953f.woff2) format(&#39;woff2&#39;),url(https://pss.bdstatic.com/r/www/cache/static/protocol/https/global/font/lama-sans/LamaSans-BlackItalic_064486f.otf)}.index_chat-body-container__Lo-B6{width:100%;overflow-y:hidden}.index_chat-body-container__Lo-B6::-webkit-scrollbar{display:none}.index_chat-body-container__Lo-B6 .index_content-area__mXkZx{-webkit-box-flex:1;-webkit-flex:1;flex:1;height:100%;will-change:scroll-position;-webkit-overflow-scrolling:touch;min-height:100%;position:relative}.index_chat-body-container__Lo-B6 .index_content-area__mXkZx .index_question-container__VKhsH{position:relative;width:100%}.index_chat-body-container__Lo-B6 .index_content-area__mXkZx .index_answer-container__K3lar{position:relative;width:100%}.index_scroll-container__Wsadr{height:100%}.index_scroll-container__Wsadr .index_content-area__mXkZx{overflow-x:hidden;overflow-y:scroll;overflow-y:overlay}a{color:#2440b3;text-decoration:none}a em{color:#f73131;text-decoration:none}a:hover{text-decoration:underline;color:#315efb}a:hover em{text-decoration:underline}a:visited{color:#771caa}a:active{color:#f73131;text-decoration:none}a:active em{text-decoration:none}em{color:#f73131}body{min-width:1116px}#content_right a{text-decoration:none}#content_right a:hover{text-decoration:underline}body.pc-fresh-wrapper-ext{min-width:1412px}body.pc-fresh-wrapper-con{min-width:1380px}.pc-fresh-title-ext #container.sam_newgrid .c-container .t a,.pc-fresh-title-ext #container.sam_newgrid .c-container .c-title a,.pc-fresh-title-ext #container.sam_newgrid a .t,.pc-fresh-title-ext #container.sam_newgrid a .c-title,.pc-fresh-title-ext #container.sam_newgrid .c-container .t a em,.pc-fresh-title-ext #container.sam_newgrid .c-container .c-title a em,.pc-fresh-title-ext #container.sam_newgrid a .t em,.pc-fresh-title-ext #container.sam_newgrid a .c-title em{text-decoration:none}.pc-fresh-title-ext #container.sam_newgrid .c-container .t a:hover,.pc-fresh-title-ext #container.sam_newgrid .c-container .c-title a:hover,.pc-fresh-title-ext #container.sam_newgrid a .t:hover,.pc-fresh-title-ext #container.sam_newgrid a .c-title:hover{text-decoration:underline;text-decoration-thickness:1px;text-underline-offset:3px;color:#2440b3}.pc-fresh-title-ext #container.sam_newgrid .c-container .t a:hover em,.pc-fresh-title-ext #container.sam_newgrid .c-container .c-title a:hover em,.pc-fresh-title-ext #container.sam_newgrid a .t:hover em,.pc-fresh-title-ext #container.sam_newgrid a .c-title:hover em{text-decoration:underline;text-decoration-thickness:1px;text-underline-offset:3px}.pc-fresh-title-ext #container.sam_newgrid .c-container .c-title-sub,.pc-fresh-title-ext #container.sam_newgrid .c-container .c-title-sub a,.pc-fresh-title-ext #container.sam_newgrid .c-container a .c-title-sub,.pc-fresh-title-ext #container.sam_newgrid .c-container .c-group-title a,.pc-fresh-title-ext #container.sam_newgrid .c-container .c-title-sub em,.pc-fresh-title-ext #container.sam_newgrid .c-container .c-title-sub a em,.pc-fresh-title-ext #container.sam_newgrid .c-container a .c-title-sub em,.pc-fresh-title-ext #container.sam_newgrid .c-container .c-group-title a em{text-decoration:none}.pc-fresh-title-ext #container.sam_newgrid .c-container .c-title-sub:hover,.pc-fresh-title-ext #container.sam_newgrid .c-container .c-title-sub a:hover,.pc-fresh-title-ext #container.sam_newgrid .c-container a .c-title-sub:hover,.pc-fresh-title-ext #container.sam_newgrid .c-container .c-group-title a:hover{text-decoration:underline;text-decoration-thickness:1px;text-underline-offset:3px;color:#2440b3}.pc-fresh-title-ext #container.sam_newgrid .c-container .c-title-sub:hover em,.pc-fresh-title-ext #container.sam_newgrid .c-container .c-title-sub a:hover em,.pc-fresh-title-ext #container.sam_newgrid .c-container a .c-title-sub:hover em,.pc-fresh-title-ext #container.sam_newgrid .c-container .c-group-title a:hover em{text-decoration:underline;text-decoration-thickness:1px;text-underline-offset:3px;color:#f73131}.pc-fresh-title-ext #container.sam_newgrid .c-showurl-hover,.pc-fresh-title-ext #container.sam_newgrid .c-showem{color:#2440b3}.pc-fresh-title-ext #container.sam_newgrid .c-showurl-hover,.pc-fresh-title-ext #container.sam_newgrid .c-showem,.pc-fresh-title-ext #container.sam_newgrid .c-showurl-hover em,.pc-fresh-title-ext #container.sam_newgrid .c-showem em{text-decoration:underline!important;text-decoration-thickness:1px!important;text-underline-offset:3px}.pc-fresh-title-ext #container.sam_newgrid .c-showurl-hover em,.pc-fresh-title-ext #container.sam_newgrid .c-showem em{color:#f73131}.pc-fresh-title-ext #container.sam_newgrid .c-showem{color:#f73131}.pc-fresh-title-ext #container.sam_newgrid .c-group-title a{color:#2440b3}.pc-fresh-title-con #container.sam_newgrid{}.pc-fresh-title-con #container.sam_newgrid .c-container .t a,.pc-fresh-title-con #container.sam_newgrid .c-container .c-title a,.pc-fresh-title-con #container.sam_newgrid a .t,.pc-fresh-title-con #container.sam_newgrid a .c-title,.pc-fresh-title-con #container.sam_newgrid .c-container .t a:hover,.pc-fresh-title-con #container.sam_newgrid .c-container .c-title a:hover,.pc-fresh-title-con #container.sam_newgrid a .t:hover,.pc-fresh-title-con #container.sam_newgrid a .c-title:hover,.pc-fresh-title-con #container.sam_newgrid .c-container .t a em,.pc-fresh-title-con #container.sam_newgrid .c-container .c-title a em,.pc-fresh-title-con #container.sam_newgrid a .t em,.pc-fresh-title-con #container.sam_newgrid a .c-title em{text-decoration-thickness:1px;text-underline-offset:3px}.pc-fresh-title-con #container.sam_newgrid .c-border .t a,.pc-fresh-title-con #container.sam_newgrid .c-border .c-title a{text-decoration:none}.pc-fresh-title-con #container.sam_newgrid .c-border .t a em,.pc-fresh-title-con #container.sam_newgrid .c-border .c-title a em{text-decoration:none}.pc-fresh-title-con #container.sam_newgrid .c-border .t a:hover,.pc-fresh-title-con #container.sam_newgrid .c-border .c-title a:hover,.pc-fresh-title-con #container.sam_newgrid .c-container .c-title-sub a:hover,.pc-fresh-title-con #container.sam_newgrid .c-container .c-group-title a:hover{text-decoration:underline;text-decoration-thickness:1px!important;text-underline-offset:3px;color:#315efb}.pc-fresh-title-con #container.sam_newgrid .c-border .t a:hover em,.pc-fresh-title-con #container.sam_newgrid .c-border .c-title a:hover em,.pc-fresh-title-con #container.sam_newgrid .c-container .c-title-sub a:hover em,.pc-fresh-title-con #container.sam_newgrid .c-container .c-group-title a:hover em{text-decoration:underline;text-decoration-thickness:1px!important;text-underline-offset:3px;color:#f73131}.pc-fresh-title-con #container.sam_newgrid .c-showurl-hover,.pc-fresh-title-con #container.sam_newgrid .c-showem{text-decoration-thickness:1px;text-underline-offset:3px}.pc-fresh-title-con #container.sam_newgrid .c-group-title a{color:#2440b3}.pc-fresh-title-con #container.sam_newgrid .c-container .t,.pc-fresh-title-ext #container.sam_newgrid .c-container .t,.pc-fresh-title-con #container.sam_newgrid .c-container .c-title,.pc-fresh-title-ext #container.sam_newgrid .c-container .c-title{font:500 18px/24px PingFang SC,Arial,Helvetica,sans-serif}.pc-fresh-title-con #container.sam_newgrid .c-container .c-group-title,.pc-fresh-title-ext #container.sam_newgrid .c-container .c-group-title{font:500 18px/24px PingFang SC,Arial,Helvetica,sans-serif}.pc-fresh-title-con #container.sam_newgrid .c-container .c-group-title .c-group-arrow-icon,.pc-fresh-title-ext #container.sam_newgrid .c-container .c-group-title .c-group-arrow-icon{display:none}.pc-fresh-title-con #container.sam_newgrid .c-container .c-title-sub,.pc-fresh-title-ext #container.sam_newgrid .c-container .c-title-sub{font:500 16px/22px PingFang SC,Arial,Helvetica,sans-serif}#container.sam_newgrid .c-container .t,#container.sam_newgrid .c-container .c-title{font-size:18px;line-height:22px}.card-visited{color:#771caa}.card-visited em{color:#f73131}.card-visited:hover{color:#771caa}#rs .new-pmd .inc-rs-new-title{line-height:14px}#rs .new-pmd .new-inc-rs-table{width:704px;border-collapse:collapse;margin-bottom:-9px}#rs .new-pmd .new-inc-rs-table td{width:16px}#rs .new-pmd .new-inc-rs-table th{width:224px;line-height:26px}#rs .new-inc-rs-item{width:224px;overflow:hidden;display:inline-block;text-overflow:ellipsis;vertical-align:top;margin-top:2px}.new-pmd .c-recommend{padding-bottom:10px}.new-pmd .c-recommend .recommend-line-height-new{line-height:1.8}.new-pmd .c-recommend .recommend-line-one{height:24px;overflow:hidden}.new-pmd .c-recommend .recommend-line-one .recommend-item-a{display:inline-block;height:24px;line-height:24px;padding:0 6px;background:#F5F5F6;border-radius:6px;text-decoration:none}.new-pmd .c-recommend .recommend-line-one .recommend-item-a:hover{background-color:#F0F0F1}.new-pmd .c-recommend .recommend-icon-bear-circle-new{width:14px;height:15px;line-height:16px;text-align:center;color:#fff;background-color:#91B9F7;margin-bottom:-6px;border-radius:4px;overflow:visible;padding-left:2px;padding-top:1px}.new-pmd .recommend-none-border{border-top:0;margin-bottom:-4px;padding-bottom:8px;border-color:#f2f2f2}.new-pmd .recommend-a-gap{padding-top:3px;padding-bottom:4px;padding-right:6px;padding-left:6px;border-radius:6px}.new-pmd .recommend-a-gap:hover{text-decoration:underline}.new-pmd .new-url-right-icon{position:relative;top:-3px;font-size:16px}.pc-aftclk-upgrade{margin-top:10px}.pc-aftclk-upgrade .recommend-tip{font-size:14px;font-weight:500;line-height:14px}.pc-aftclk-upgrade .c-recommend .recommend-line-one{height:28px}.pc-aftclk-upgrade .c-recommend .recommend-line-one .recommend-item-a{height:28px;line-height:28px;padding:0 9px;font-size:14px;display:inline-flex;align-items:center;font-weight:var(--cos-font-medium,500);color:var(--cos-color-text,#333);background:var(--cos-color-bg-dent,#f5f6f9)}.pc-aftclk-upgrade .c-recommend .recommend-line-one .recommend-item-a .aftclk-icon{margin-right:6px}.pc-aftclk-upgrade .c-recommend .recommend-line-one .recommend-item-a:hover{background-color:#4E6EF20F;color:var(--cos-color-text-primary,#4e6ef2);font-weight:400}.pc-aftclk-upgrade .c-recommend .recommend-line-one .recommend-item-a:hover .aftclk-icon{color:var(--cos-color-text-primary,#4e6ef2);background-image:url(https://gips0.baidu.com/it/u=837208242,3555265853&amp;fm=3028&amp;app=3028&amp;f=PNG&amp;fmt=auto&amp;q=75&amp;size=f42_42)}.pc-aftclk-upgrade .c-gap-top-middle{margin-top:9px}.pc-aftclk-upgrade .aftclk-icon{display:inline-block;background-image:url(https://psstatic.cdn.bcebos.com/basics/recommend/rs-icon_1730789535000.svg);background-size:100% 100%;background-repeat:no-repeat;background-position:0 0;width:14px;height:14px;margin-right:8px}.pc-aftclk-upgrade .recommend-none-border{border:0}.selected-search-box{z-index:300;position:absolute;cursor:pointer;border:0;background:#FFF;box-shadow:0 2px 10px 0 rgba(0,0,0,.1);border-radius:6px;padding:10px 15px 9px 16px}.selected-search-box a,.selected-search-box a:hover,.selected-search-box a:visited{text-decoration:none;color:#333;line-height:13px;height:13px;overflow:hidden}.selected-search-box i{float:left;margin-left:8px;color:#4E6EF2;font-size:14px;width:14px;height:14px;vertical-align:middle;font-weight:bolder}.selected-search-box span{padding-top:20px;margin-top:-20px;overflow:hidden;float:left;font-family:Arial,MicrosoftYaHei;font-size:13px;line-height:13px;max-width:156px;white-space:nowrap;text-overflow:ellipsis;vertical-align:text-bottom}.news-readed .tts-title{color:#771CAA}.news-readed .tts-title a{color:#771CAA}.news-reading .tts-title{color:#315efb}.news-reading .tts-title a{color:#315efb}.open-result-tts .tts{display:inline-block}.darkmode .news-readed .tts-title{color:#EA80FF}.darkmode .news-readed .tts-title a{color:#EA80FF}.darkmode .news-reading .tts-title{color:#FFF762}.darkmode .news-reading .tts-title a{color:#FFF762}.toast-for-result{position:fixed;top:50%;left:50%;height:30px;transform:translate(-50%,-50%);padding:0 16px;line-height:30px;border-radius:15px;box-shadow:0 1px 5px rgba(0,0,0,.1);background:#626675;color:#FFF}#searchTag.tag-fixed{position:fixed;top:70px;box-shadow:0 12px 10px -10px rgba(0,0,0,.1);padding-top:1px;margin-top:-1px;padding-bottom:16px}.new_input_wrapper #searchTag.tag-fixed{top:74px;padding-bottom:15px}.wrapper_new #head.no-box-shadow,.wrapper_new #head.no-box-shadow.s_down{box-shadow:none}.guide-info-new{cursor:pointer;z-index:999;height:34px;padding:0 15px;min-width:120px;background-color:rgba(98,102,117,.8);box-shadow:0 2px 10px 0 rgba(0,0,0,.1);border-radius:6px;text-align:left;position:absolute;line-height:35px;white-space:nowrap}.guide-close{color:#D7D9E0;margin-left:8px;display:inline-block!important;height:34px;text-align:center;vertical-align:top;font-size:13px!important}.guide-close:hover{color:#fff!important}.guide-arrow-bottom{top:-11px;right:10px;background:url(https://pss.bdstatic.com/r/www/cache/static/protocol/https/global/img/arrow-bottom_a44a0c6.png) no-repeat 0 0}.guide-arrow-bottom{position:absolute;opacity:.8;height:11px;width:11px;background-size:11px 11px}.guide-text{display:inline-block;vertical-align:top;font-size:13px;font-family:Arial,sans-serif;color:#fff;margin-right:-5px}.color222{color:#222}.no-outline-while-click:not(.exclude-tabindex){outline:0}.s-tipbox{color:#333;text-align:center}.s-tipbox .s-tipbox-mask{position:fixed;top:0;bottom:0;left:0;right:0;background:rgba(0,0,0,.5);z-index:999}.s-tipbox .s-tipbox-con{position:fixed;top:50%;left:50%;width:290px;height:353px;margin-left:-145px;margin-top:-177px;background:#fff;border-radius:16px;z-index:999}.s-tipbox .s-tipbox-con .s-tipbox-close{position:absolute;right:7px;top:7px;width:40px;height:40px;padding:9px;box-sizing:border-box;background-image:url(https://pss.bdstatic.com/r/www/cache/static/protocol/https/tipbox/img/close-btn_364ba48.png);background-position:center;background-repeat:no-repeat;background-size:22px;cursor:pointer}.s-tipbox .s-tipbox-con .s-tipbox-img{width:290px;height:117px}.s-tipbox .s-tipbox-con .s-tipbox-top{padding:30px 0;padding-bottom:24px}.s-tipbox .s-tipbox-con .s-tipbox-top h3{font-family:PingFangSC-Medium;font-size:23px;color:#333;letter-spacing:0;text-align:center;line-height:20px;height:20px}.s-tipbox .s-tipbox-con .s-tipbox-top p{font-family:PingFangSC-Regular;font-size:14px;color:#666;letter-spacing:.5px;line-height:23px;width:190px;margin:auto;margin-top:16px;text-align:left}.s-tipbox .s-tipbox-con .s-tipbox-btn-sure{width:190px;height:46px;line-height:46px;text-align:center;background:#38F;border-radius:100px;margin:auto;font-family:PingFangSC-Medium;font-size:16px;color:#fff;cursor:pointer}.s-tipbox .s-tipbox-con .s-tipbox-btn-cancel{font-family:PingFangSC-Regular;font-size:14px;color:#666;text-align:center;line-height:14px;text-decoration:underline;margin-top:20px;cursor:pointer}.bds-lead-img{width:100%;height:100%}.s-banner-close{position:absolute;right:7px;top:7px;width:40px;height:40px;padding:9px;box-sizing:border-box;background-image:url(https://pss.bdstatic.com/r/www/cache/static/protocol/https/tipbox/img/close-btn_364ba48.png);background-position:center;background-repeat:no-repeat;background-size:22px;cursor:pointer}.search-quit-dialog-wrap{display:none;position:fixed;z-index:302;left:0;top:0;width:100%;height:100%}.search-quit-dialog-wrap .mask{position:absolute;top:0;left:0;width:100%;height:100%;background:#000;opacity:.5}.search-quit-dialog-wrap .popup-box{position:absolute;box-sizing:border-box;top:50%;left:50%;width:416px;height:166px;padding:24px;margin-left:-208px;margin-top:-83px;background:#FFF;box-shadow:0 2px 10px 0 rgba(0,0,0,.1);border-radius:12px}.search-quit-dialog-wrap .head{height:20px;margin-top:1px;font-size:20px;color:#222;line-height:20px;font-weight:400}.search-quit-dialog-wrap .head span{position:absolute;top:16px;right:16px;color:#d7d9e0;cursor:pointer;font-family:cIconfont!important;font-style:normal;font-size:16px;-webkit-font-smoothing:antialiased}.search-quit-dialog-wrap .head span:hover{color:#315efb}.search-quit-dialog-wrap .body{height:13px;margin:16px 0 39px;font-size:13px;color:#626675;line-height:13px;font-weight:400}.search-quit-dialog-wrap .bottom{font-size:0;text-align:right}.search-quit-dialog-wrap .bottom .exit:active{box-shadow:none}.search-quit-dialog-wrap .bottom .exit:hover{background-color:#315efb;color:#fff!important}#seth{display:inline;behavior:url(#default#homepage)}#setf{display:inline}#sekj{margin-left:14px}#st,#sekj{display:none}.s_ipt_wr{border:1px solid #b6b6b6;border-color:#7b7b7b #b6b6b6 #b6b6b6 #7b7b7b;background:#fff;display:inline-block;vertical-align:top;width:539px;margin-right:0;border-right-width:0;border-color:#b8b8b8 transparent #ccc #b8b8b8;overflow:hidden}.sam_search.s_ipt_wr{border-color:#1D4FFF!important}.wrapper_s .s_ipt_wr{width:478px}.wrapper_s .s_ipt{width:357px}.wrapper_s .s_ipt_tip{width:357px}.sam_search.s_ipt_wr:hover,.sam_search.s_ipt_wr.ipthover{border:color #1D4FFF!important}.sam_search.s_ipt_wr.iptfocus{border-color:#1D4FFF}.s_ipt_wr:hover,.s_ipt_wr.ipthover{border-color:#999 transparent #b3b3b3 #999}.s_ipt_wr.iptfocus{border-color:#4791ff transparent #4791ff #4791ff}.s_ipt_tip{color:#aaa;position:absolute;z-index:-10;font:16px/22px arial;height:32px;line-height:32px;padding-left:7px;overflow:hidden;width:526px}.s_ipt{width:526px;height:22px;font:16px/18px arial;line-height:22px;margin:6px 0 0 7px;padding:0;background:transparent;border:0;outline:0;-webkit-appearance:none}#kw{position:relative}#u .username i{background-position:-408px -144px}.bdpfmenu,.usermenu{border:1px solid #d1d1d1;position:absolute;width:105px;top:36px;z-index:302;box-shadow:1px 1px 5px #d1d1d1;-webkit-box-shadow:1px 1px 5px #d1d1d1;-moz-box-shadow:1px 1px 5px #d1d1d1;-o-box-shadow:1px 1px 5px #d1d1d1}.bdpfmenu{font-size:12px;background-color:#fff}.bdpfmenu a,.usermenu a{display:block;text-align:left;padding:0 9px;line-height:26px;text-decoration:none}.briiconsbg{background-repeat:no-repeat;background-size:300px 18px;background-image:url(https://pss.bdstatic.com/r/www/cache/static/protocol/https/home/img/icons_0c37e9b.png);background-image:url(https://pss.bdstatic.com/r/www/cache/static/protocol/https/home/img/icons_809ae65.gif)\9}#u{z-index:301;position:absolute;right:0;top:0;margin:21px 9px 5px 0;padding:0}.wrapper_s #u{margin-right:3px}#u a{text-decoration:underline;color:#333;margin:0 7px}.wrapper_s #u a{margin-right:0 6px}#u div a{text-decoration:none}#u a:hover{text-decoration:underline}#u .back_org{color:#666;float:left;display:inline-block;height:24px;line-height:24px}#u .bri{display:inline-block;width:24px;height:24px;float:left;line-height:24px;color:transparent;background:url(https://pss.bdstatic.com/r/www/cache/static/protocol/https/home/img/icons_0c37e9b.png) no-repeat 4px 3px;background-size:300px 18px;background-image:url(https://pss.bdstatic.com/r/www/cache/static/protocol/https/home/img/icons_809ae65.gif)\9;overflow:hidden}#u .bri:hover,#u .bri.brihover{background-position:-18px 3px}#mCon #imeSIcon{background-position:-408px -144px;margin-left:0}#mCon span{color:#333}.bdpfmenu a:link,.bdpfmenu a:visited,#u .usermenu a:link,#u .usermenu a:visited{background:#fff;color:#333}.bdpfmenu a:hover,.bdpfmenu a:active,#u .usermenu a:hover,#u .usermenu a:active{background:#38f;text-decoration:none;color:#fff}.bdpfmenu{width:70px}.usermenu{width:68px;right:8px}#wrapper .bdnuarrow{width:0;height:0;font-size:0;line-height:0;display:block;position:absolute;top:-10px;left:50%;margin-left:-5px}#wrapper .bdnuarrow em,#wrapper .bdnuarrow i{width:0;height:0;font-size:0;line-height:0;display:block;position:absolute;border:5px solid transparent;border-style:dashed dashed solid}#wrapper .bdnuarrow em{border-bottom-color:#d8d8d8;top:-1px}#wrapper .bdnuarrow i{border-bottom-color:#fff;top:0}#prefpanel{background:#fafafa;display:none;opacity:0;position:fixed;_position:absolute;top:-359px;z-index:500;width:100%;min-width:960px;border-bottom:1px solid #ebebeb}#prefpanel form{_width:850px}#kw_tip{cursor:default;display:none;margin-top:1px}#bds-message-wrapper{top:43px}.quickdelete-wrap{position:relative}.quickdelete-wrap input,.wrapper_l .quickdelete-wrap input{width:500px}.wrapper_s .quickdelete-wrap input{width:402px}input::-ms-clear{display:none}.quickdelete{width:18px;height:18px;font-size:16px;line-height:18px;text-align:center;position:absolute;display:none;top:50%;right:16px;margin-top:-9px;cursor:pointer}.quickdelete-line{display:none;height:14px;width:1px;background-color:#F5F5F6;position:absolute;top:50%;right:0;margin-top:-7px}#form.has-voice.fm .quickdelete{right:95px}#form.has-voice.fm .quickdelete-line{right:83px}#form.has-soutu .quickdelete{right:63px}#form.has-soutu .quickdelete-line{right:51px}#form.sam_search .quickdelete{color:#E4E4E5}#form.sam_search .quickdelete:hover{color:#C4C7CE}#form.sam_search .quickdelete-line{height:24px;margin-top:-12px}#form.has-voice.sam_search.fm .quickdelete{right:111px}#form.has-voice.sam_search.fm .quickdelete-line{right:95px}#form.has-soutu.sam_search .quickdelete{right:71px}#form.has-soutu.sam_search .quickdelete-line{right:55px}#lh a{margin-left:25px}.bdbriwrapper-tuiguang{display:none!important}.soutu-input{padding-left:55px!important}.soutu-input-image{position:absolute;left:1px;top:1px;height:28px;width:49px;z-index:1;padding:0;background:#e6e6e6;border:1px solid #e6e6e6}.soutu-input-thumb{height:28px;width:28px;min-width:1px}.soutu-input-close{position:absolute;right:0;top:0;cursor:pointer;display:block;width:22px;height:28px}.soutu-input-close::after{content:&quot; &quot;;position:absolute;right:3px;top:50%;cursor:pointer;margin-top:-7px;display:block;width:14px;height:14px;background:url(https://pss.bdstatic.com/r/www/cache/static/protocol/https/soutu/img/soutu_icons_new_8abaf8a.png) no-repeat -163px 0}.soutu-input-image:hover .soutu-input-close::after{background-position:-215px 2px}.darkmode.wrapper_new #s_tab .cur-tab,.darkmode.wrapper_new #s_tab a,.darkmode.wrapper_new #u&gt;a{color:#FFD862}.darkmode.wrapper_new #s_tab .cur-tab:hover,.darkmode.wrapper_new #s_tab a:hover,.darkmode.wrapper_new #u&gt;a:hover{color:#FFF762}.darkmode.wrapper_new #u .s-top-img-wrapper{border:1px solid #FFD862}.darkmode.wrapper_new .c-showurl-hover{color:#FFF762}.darkmode.wrapper_new #page{background:0 0}.darkmode.wrapper_new #s_tab .cur-tab:after{background:#C4C7CE}.darkmode.wrapper_new #s_tab .cur-tab:before,.darkmode.wrapper_new #s_tab .s-tab-item:before{color:#A8ACAD}.darkmode.wrapper_new .search-setting .prefpanelrestore{color:#333!important;background-color:#F5F5F6!important}.darkmode #rs,.darkmode #head,.darkmode #foot{background:0 0}.darkmode a{color:#FFD862}.darkmode a:hover{color:#FFF762}.darkmode a:visited{color:#E7BDFF}.darkmode.dark #head{background:#1F1F25}.darkmode.blue #head{background:#141E42}.darkmode #container.sam_newgrid .result .c-tools .c-icon,.darkmode #container.sam_newgrid .result-op .c-tools .c-icon{color:#A8ACAD}.darkmode .index-logo-peak{display:block}.darkmode .index-logo-src,.darkmode .index-logo-srcnew{display:none}.aging-tools-gap #head{position:absolute}.aging-tools-gap #head.s_down{position:fixed;top:0}.fb-hint{margin-top:5px;transition-duration:.9s;opacity:0;display:none;color:red}.fb-img{display:none}.fb-hint-tip{height:44px;line-height:24px;background-color:#38f;color:#fff;box-sizing:border-box;width:269px;font-size:16px;padding:10px;padding-left:14px;position:absolute;top:-65px;right:-15px;border-radius:3px;z-index:299}.fb-hint-tip::before{content:&quot;&quot;;width:0;height:0;display:block;position:absolute;border-left:8px solid transparent;border-right:8px solid transparent;border-top:8px solid #38f;bottom:-8px;right:25px}.fb-mask,.fb-mask-light{position:fixed;top:0;left:0;bottom:0;right:0;z-index:296;background-color:#000;filter:alpha(opacity=60);background-color:rgba(0,0,0,.6)}.fb-mask-light{background-color:#fff;filter:alpha(opacity=0);background-color:rgba(255,255,255,0)}.fb-success .fb-success-text{text-align:center;color:#333;font-size:13px;margin-bottom:14px}.fb-success-text.fb-success-text-title{color:#3b6;font-size:16px;margin-bottom:16px}.fb-success-text-title i{width:16px;height:16px;margin-right:5px}.fb-list-container{box-sizing:border-box;padding:4px 8px;position:absolute;top:0;left:0;bottom:0;right:0;z-index:298;display:block;width:100%;cursor:pointer;margin-top:-5px;margin-left:-5px}.fb-list-container-hover{background-color:#fff;border:2px #38f solid}.fb-list-container-first{box-sizing:border-box;padding-left:10px;padding-top:5px;position:absolute;top:0;left:0;bottom:0;right:0;z-index:297;display:block;width:100%;cursor:pointer;margin-top:-5px;margin-left:-5px;border:3px #f5f5f5 dashed;border-radius:3px}.fb-des-content{font-size:13px!important;color:#000}.fb-des-content::-webkit-input-placeholder{font-size:13px!important;color:#9a9a9a}.fb-des-content:-moz-placeholder{font-size:13px!important;color:#9a9a9a}.fb-des-content::-moz-placeholder{font-size:13px!important;color:#9a9a9a}.fb-des-content:-ms-input-placeholder{font-size:13px!important;color:#9a9a9a}.fb-btn,.fb-btn:visited{color:#333!important}.fb-select{position:relative;background-color:#fff;border:1px solid #ccc}.fb-select i{position:absolute;right:2px;top:7px}.fb-type{width:350px;box-sizing:border-box;height:28px;font-size:13px;line-height:28px;border:0;word-break:normal;word-wrap:normal;position:relative;appearance:none;-moz-appearance:none;-webkit-appearance:none;display:inline-block;vertical-align:middle;line-height:normal;color:#333;background-color:transparent;border-radius:0;overflow:hidden;outline:0;padding-left:5px}.fb-type::-ms-expand{display:none}.fb-btn{display:inline-block;padding:0 14px;margin:0;height:24px;line-height:25px;font-size:13px;filter:chroma(color=#000000);*zoom:1;border:1px solid #d8d8d8;cursor:pointer;font-family:inherit;font-weight:400;text-align:center;vertical-align:middle;background-color:#f9f9f9;overflow:hidden;outline:0}.fb-btn:hover{border-color:#388bff}.fb-btn:active{border-color:#a2a6ab;background-color:#f0f0f0;box-shadow:inset 1px 1px 1px #c7c7c7;-webkit-box-shadow:inset 1px 1px 1px #c7c7c7;-moz-box-shadow:inset 1px 1px 1px #c7c7c7;-o-box-shadow:inset 1px 1px 1px #c7c7c7}a.fb-btn{text-decoration:none}button.fb-btn{height:26px;_line-height:18px;*overflow:visible}button.fb-btn::-moz-focus-inner{padding:0;border:0}.fb-btn .c-icon{margin-top:5px}.fb-btn-primary,.fb-btn-primary:visited{color:#fff!important}.fb-btn-primary{background-color:#388bff;_width:82px;border-color:#3c8dff #408ffe #3680e6}.fb-btn-primary:hover{border-color:#2678ec #2575e7 #1c6fe2 #2677e7;background-color:#388bff;background-image:url(data:image/png;    base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAACCAMAAACuX0YVAAAABlBMVEVnpv85i/9PO5r4AAAAD0lEQVR42gEEAPv/AAAAAQAFAAIros7PAAAAAElFTkSuQmCC);background-repeat:repeat-x;box-shadow:1px 1px 1px rgba(0,0,0,.4);-webkit-box-shadow:1px 1px 1px rgba(0,0,0,.4);-moz-box-shadow:1px 1px 1px rgba(0,0,0,.4);-o-box-shadow:1px 1px 1px rgba(0,0,0,.4)}.fb-btn-primary:active{border-color:#178ee3 #1784d0 #177bbf #1780ca;background-color:#388bff;background-image:none;box-shadow:inset 1px 1px 1px rgba(0,0,0,.15);-webkit-box-shadow:inset 1px 1px 1px rgba(0,0,0,.15);-moz-box-shadow:inset 1px 1px 1px rgba(0,0,0,.15);-o-box-shadow:inset 1px 1px 1px rgba(0,0,0,.15)}.fb-feedback-right-dialog{position:fixed;z-index:299;bottom:0;right:0}.fb-feedback-list-dialog,.fb-feedback-list-dialog-left{position:absolute;z-index:299}.fb-feedback-list-dialog:before{content:&quot;&quot;;width:0;height:0;display:block;position:absolute;top:15px;left:-6px;border-top:8px solid transparent;border-bottom:8px solid transparent;border-right:8px solid #fff}.fb-feedback-list-dialog-left:before{content:&quot;&quot;;width:0;height:0;display:block;position:absolute;top:15px;right:-6px;border-top:8px solid transparent;border-bottom:8px solid transparent;border-left:8px solid #fff}.fb-header{padding-left:20px;padding-right:20px;margin-top:14px;text-align:left;-moz-user-select:none}.fb-header .fb-close{color:#e0e0e0}.fb-close{text-decoration:none;margin-top:2px;float:right;font-size:20px;font-weight:700;line-height:18px;color:#666;text-shadow:0 1px 0 #fff}.fb-photo-block{display:none}.fb-photo-block-title{font-size:13px;color:#333;padding-top:10px}.fb-photo-block-title-span{color:#999}.fb-photo-sub-block{margin-top:10px;margin-bottom:10px;width:60px;text-align:center}.fb-photo-sub-block-hide{display:none}.fb-photo-update-block{overflow:hidden}.fb-photo-update-item-block{width:100px;height:100px;background:red;border:solid 1px #ccc;margin-top:10px;float:left;margin-right:20px;position:relative;background:url(https://pss.bdstatic.com/r/www/cache/static/protocol/https/global/img/feedback_add_photo_69ff822.png);background-repeat:no-repeat;background-size:contain;background-position:center center;background-size:24px 24px}.fb-photo-block-title-ex{font-size:13px;float:right}.fb-photo-block-title-ex img{vertical-align:text-top;margin-right:4px}.fb-photo-block-title-span{margin-left:4px;color:#999}.fb-photo-update-item-show-img{width:100%;height:100%;display:none}.fb-photo-update-item-close{width:13px;height:13px;position:absolute;top:-6px;right:-6px;display:none}.fb-photo-block input{display:none}.fb-photo-update-hide{display:none}.fb-photo-update-item-block{width:60px;height:60px;border:solid 1px #ccc;float:left}.fb-photo-block-example{position:absolute;top:0;left:0;display:none;background-color:#fff;padding:14px;padding-top:0;width:392px}.fb-photo-block-example-header{padding-top:14px;overflow:hidden}.fb-photo-block-example-header p{float:left}.fb-photo-block-example-header img{float:right;width:13px;height:13px}.fb-photo-block-example-img img{margin:0 auto;margin-top:14px;display:block;width:200px}.fb-photo-block-example-title{text-align:center}.fb-photo-block-example-title-big{font-size:14px;color:#333}.fb-photo-block-example-title-small{font-size:13px;color:#666}.fb-header a.fb-close:hover{text-decoration:none}.fb-photo-block-upinfo{width:100%}.fb-header-tips{font-size:16px;margin:0;color:#333;text-rendering:optimizelegibility}.fb-body{margin-bottom:0;padding:20px;padding-top:10px;overflow:hidden;text-align:left}.fb-modal,.fb-success,.fb-vertify{background-color:#fff;cursor:default;top:100%;left:100%;width:390px;overflow:hidden;border:1px solid #999;*border:1px solid #ddd;font-size:13px;line-height:1.54}.fb-textarea textarea{width:350px;height:64px;padding:4px;margin:10px 0;vertical-align:top;resize:none;overflow:auto;box-sizing:border-box;display:inline-block;border:1px solid #ccc;-webkit-border-radius:0;-moz-border-radius:0;border-radius:0;-webkit-box-shadow:inset 0 1px 1px rgba(0,0,0,.075);-moz-box-shadow:inset 0 1px 1px rgba(0,0,0,.075);box-shadow:inset 0 1px 1px rgba(0,0,0,.075);-webkit-transition:border linear .2s,box-shadow linear .2s;-moz-transition:border linear .2s,box-shadow linear .2s;-ms-transition:border linear .2s,box-shadow linear .2s;-o-transition:border linear .2s,box-shadow linear .2s;transition:border linear .2s,box-shadow linear .2s}.fb-selected{display:none;width:12px;height:12px;background:url(data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAcAAAAFCAYAAACJmvbYAAAABGdBTUEAALGPC/xhBQAAACBjSFJNAAB6JgAAgIQAAPoAAACA6AAAdTAAAOpgAAA6mAAAF3CculE8AAAABmJLR0QAAAAAAAD5Q7t/AAAACXBIWXMAABYlAAAWJQFJUiTwAAAAJklEQVQI12NgwAEsuv/8xy9h3vX7P6oEKp/BHCqA0yhzdB0MDAwAFXkTK5la4mAAAAAASUVORK5CYII=) no-repeat 2px 3px}.fb-guide{padding-top:10px;color:#9a9a9a;margin-left:-20px;padding-left:20px;border-right-width:0;margin-right:-20px;padding-right:25px;margin-bottom:-20px;padding-bottom:15px}.fb-footer{padding-top:10px;text-align:left}.fb-block{overflow:hidden;position:relative}.fb-block .fb-email{height:28px;line-height:26px;width:350px;border:1px solid #ccc;padding:4px;padding-top:0;box-sizing:border-box;padding-bottom:0;display:inline-block;font-family:&#39;Helvetica Neue&#39;,Helvetica,Arial,sans-serif;vertical-align:middle!important;-webkit-border-radius:0;-moz-border-radius:0;border-radius:0;-webkit-box-shadow:inset 0 1px 1px rgba(0,0,0,.075);-moz-box-shadow:inset 0 1px 1px rgba(0,0,0,.075);box-shadow:inset 0 1px 1px rgba(0,0,0,.075);-webkit-transition:border linear .2s,box-shadow linear .2s;-moz-transition:border linear .2s,box-shadow linear .2s;-ms-transition:border linear .2s,box-shadow linear .2s;-o-transition:border linear .2s,box-shadow linear .2s;transition:border linear .2s,box-shadow linear .2s}.fb-email{font-size:13px!important;color:#000}.fb-email::-webkit-input-placeholder{font-size:13px!important;color:#9a9a9a}.fb-email:-moz-placeholder{font-size:13px!important;color:#9a9a9a}.fb-email::-moz-placeholder{font-size:13px!important;color:#9a9a9a}.fb-email:-ms-input-placeholder{font-size:13px!important;color:#9a9a9a}.fb-cut-block{height:15px;padding-bottom:10px}.fb-canvas-block{height:172px;border:1px solid #ccc;margin-bottom:10px;position:relative;overflow:hidden;width:100%;background-position:center;box-sizing:border-box}.fb-canvas-block img{width:350px;position:absolute}.fb-canvas-block img[src=&quot;&quot;]{opacity:0}.fb-cut-input{width:14px;height:14px;margin:0;margin-right:10px;display:inline-block;border:1px solid #ccc}.fb-cut-btn{width:60px!important}#fb_tips_span{vertical-align:middle}#fb_popwindow{display:block;left:457px;top:69.5px;position:absolute;width:450px;z-index:999999;background:none repeat scroll 0 0 #fff;border:1px solid #999;border-radius:3px;box-shadow:0 0 9px #999;padding:0}#feedback_dialog_content{text-align:center}#fb_right_post_save:hover{background-image:url(data:image/png;    base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAACCAMAAACuX0YVAAAABlBMVEVnpv85i/9PO5r4AAAAD0lEQVR42gEEAPv/AAAAAQAFAAIros7PAAAAAElFTkSuQmCC);background-repeat:repeat-x;box-shadow:1px 1px 1px rgba(0,0,0,.4);-webkit-box-shadow:1px 1px 1px rgba(0,0,0,.4);-moz-box-shadow:1px 1px 1px rgba(0,0,0,.4);-o-box-shadow:1px 1px 1px rgba(0,0,0,.4)}.fb-select-icon{position:absolute;bottom:6px;right:5px;width:16px;height:16px;box-sizing:content-box;background-position:center center;background-repeat:no-repeat;background-size:7px 4px;-webkit-background-size:7px 4px;background-image:url(data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAcAAAAECAYAAABCxiV9AAAABGdBTUEAALGPC/xhBQAAACBjSFJNAAB6JgAAgIQAAPoAAACA6AAAdTAAAOpgAAA6mAAAF3CculE8AAAABmJLR0QAAAAAAAD5Q7t/AAAACXBIWXMAAAsSAAALEgHS3X78AAAAKElEQVQI12Ps7Or6z4ADMDIwMDBgU1BeVsbICOMgKygvK2PEMAbdBAAhxA08t5Q3VgAAAABJRU5ErkJggg==)}.fb-select-shorter{position:relative;min-height:28px}.fb-type-container{line-height:28px;position:absolute;top:28px;width:100%;background-color:#fff;border:1px solid #ccc;z-index:300;margin-left:-1px;display:none}.fb-type-item,.fb-type-selected{height:28px;line-height:30px;padding-left:4px}.fb-type-item:hover{background:#f5F5F5}.fb-checkbox{position:relative;border-bottom:1px solid #eee;height:34px;line-height:35px}.fb-checkbox:last-child{border-bottom:0}.fb-list-wrapper{margin-top:-10px;max-height:370px;overflow-y:scroll}.fb-list-wrapper::-webkit-scrollbar{background:#fff;width:10px;height:1px}.fb-list-wrapper::-webkit-scrollbar-thumb{background-color:#e0e0e0;border-radius:10px}.fb-textarea-sug textarea{margin-top:0}&lt;/style&gt;
</textarea><textarea id="s_index_off_css" style="display:none;"><style data-for="result" id="css_result" type="text/css">#ftCon{display:none}
#qrcode{display:none}
#pad-version{display:none}
#index_guide{display:none}
#index_logo{display:none}
#u1{display:none}
.s-top-left{display:none}
.s_ipt_wr{height:32px}
body{padding:0}
#head .c-icon-bear-round{display:none}
.index_tab_top{display:none}
.index_tab_bottom{display:none}
#lg{display:none}
#m{display:none}
#ftCon{display:none}
#bottom_layer,#bottom_space,#s_wrap{display:none}
.s-isindex-wrap{display:none}
#nv{display:none!important}
#head .head_wrapper{display:block;padding-top:0!important}
.s-bottom-ctner{display:none!important}
#head .s-upfunc-menus{display:none}
#s_skin_upload{display:none}
#s_new_search_guide{display:none}</style></textarea><div id="wrapper" class="wrapper_new   "><div class='s-skin-container s-isindex-wrap skin-gray-event' ></div><div id="head" class=""><div id="s_top_wrap" class="s-top-wrap s-isindex-wrap "><div class="s-top-nav"></div><div class="s-center-box"></div></div><div id="s-top-left" class="s-top-left-new s-isindex-wrap"><a href="http://news.baidu.com" target="_blank"  class="mnav c-font-normal c-color-t">
                    
                        新闻
                    
                </a><a href="https://www.hao123.com?src=from_pc_logon" target="_blank"  class="mnav c-font-normal c-color-t">
                    
                        hao123
                    
                </a><a href="http://map.baidu.com" target="_blank"  class="mnav c-font-normal c-color-t">
                    
                        地图
                    
                </a><a href="http://tieba.baidu.com/" target="_blank"  class="mnav c-font-normal c-color-t">
                    
                        贴吧
                    
                </a><a href="https://haokan.baidu.com/?sfrom=baidu-top" target="_blank"  class="mnav c-font-normal c-color-t">
                    
                        视频
                    
                </a><a href="http://image.baidu.com/" target="_blank"  class="mnav c-font-normal c-color-t">
                    
                        图片
                    
                </a><a href="https://pan.baidu.com?from=1026962h" target="_blank"  class="mnav c-font-normal c-color-t">
                    
                        网盘
                    
                </a><a href="https://wenku.baidu.com/?fr=bdpcindex" target="_blank"  class="mnav c-font-normal c-color-t">
                    
                        文库
                    
                </a><a href="https://chat.baidu.com/search?isShowHello=1&amp;pd=csaitab&amp;setype=csaitab&amp;extParamsJson=%7B%22enter_type%22%3A%22home_tab%22%7D" target="_blank" id="csaitab" class="mnav c-font-normal c-color-t"><img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAE4AAAAqCAMAAAAqEZ1jAAAAAXNSR0IArs4c6QAAAAlwSFlzAAAhOAAAITgBRZYxYAAAAKJQTFRFAAAAenb/RpP/k2j/XXf7cXX6P5z7i2n8W3j7ZHP7jmn9VID7Qpf8jWn8Xnr6iWr8YHb7i2n8RJb9dWz8WXr8mWj9RpP8W3n8bW/8l2j+QJz9Q5f9ToT8fWv9ZW/8lWj9VID8YnX7Pp79QZr9pWb+RJT8m2f9k2j9SYz8jWn9hmr8UIT8fmv8VX78Xnn7d2z8ZHT7WHr7bm77Xnb7Z3D7YHH7RJOQRAAAACJ0Uk5TABAgICAwQEBAWF5gZXBwgICbn5+fo7+/vsLP39/f3urv73XwOA8AAAKfSURBVHja7dbJcuIwFIXhIzCxMTMNcdwMDoMZY4NxeP9X66srEckYQlPVvcvPBhZ8dVSIAvz03xNV/Luq4Xq9Hgo8nfDqjXq97hS1yXZNhXg2Z3o+f1KnLqx62y17Pp5smp+158EUaW6I52rkxLF3eoNpt1Ne+bR+Fd8U5Oc8Px6P5M0Evgq11ytpy0jgbk6e5cSxd2pb7yKOikpTouWyhbsNMsnpefZpOzsCI7c8brkM74/LqFSC7HkwVTu9lsB14ZJy745LlDdV5+3iQTXC7s9z3pNEggOh9s3woOFqJT2BmzUSLnMQEEeeh2+rkraiOrjZeL+XXEBwyp71Ybg+hWL+iltGt8ftKQI9QMzSlDxz9VofssIMdxitdJEvbo3jxqAGKXtt6N4OkpvgkmiFa50Ch1UU8zab/Ya4Br/IUnnewHCHg+FEh7/ExUIXdq+KG4N7z3hfzXDkaa5FmGpdMENrYWWjaoDrJtJLu4ajFFfbUduryN1ue4brb2KpjSvgvETe53R24U6Ggz+RYJnsCTMujmMJ9qELErkv9y7cyXAGtNCoI2Dqx9ymAl1bcYMLJ2NO5fZ2dmHLxiBGrJlxEAl7M1HkTFUzseeiWHOxYK+CrwLltRX3qTk7/4PbTXDdaBHLxytMdcllaaA56U1RSEyU56M0jotfYBLvynM091ni0GGtPO73nLkR7AaK6yqOKnG1g+wXrnqZz9nrw87bszdjjn/dDGffbgdX9YmT4HykaoLTp/UABGfyypx7a1yFNb0wpl7BddW8geSO5JU5TE63xhmOPb2uok8reN1Nrn4qjQNBlmfdvkB5beaoMgdP4LpFgbNuX3svtSwABoZ7WPOKa0Inxgl7DurMdfFXVV7savjK8WQ1/cz5+Q9e7A/jUZeiPQO0fwAAAABJRU5ErkJggg==" class="s-top-tab-image" /></a><div class="mnav s-top-more-btn"><a href="http://www.baidu.com/more/" name="tj_briicon" class="s-bri c-font-normal c-color-t" target="_blank">更多</a><div class='s-top-more' id="s-top-more"><div class='s-top-more-content clearfix'><a class="img-spacing" href='http://fanyi.baidu.com/' target='_blank' name='tj_fanyi'><img src='https://pss.bdstatic.com/static/superman/img/topnav/newfanyi-da0cea8f7e.png'/><div class="s-top-more-title c-font-normal c-color-t">翻译</div></a></div><div class='s-top-more-content clearfix'><a class="img-spacing" href='http://xueshu.baidu.com/' target='_blank' name='tj_xueshu'><img src='https://pss.bdstatic.com/static/superman/img/topnav/newxueshuicon-a5314d5c83.png'/><div class="s-top-more-title c-font-normal c-color-t">学术</div></a></div><div class='s-top-more-content clearfix'><a class="" href='https://baike.baidu.com' target='_blank' name='tj_baike'><img src='https://pss.bdstatic.com/static/superman/img/topnav/newbaike-889054f349.png'/><div class="s-top-more-title c-font-normal c-color-t">百科</div></a></div><div class='s-top-more-content clearfix'><a class="img-spacing" href='https://zhidao.baidu.com' target='_blank' name='tj_zhidao'><img src='https://pss.bdstatic.com/static/superman/img/topnav/newzhidao-da1cf444b0.png'/><div class="s-top-more-title c-font-normal c-color-t">知道</div></a></div><div class='s-top-more-content clearfix'><a class="img-spacing" href='https://jiankang.baidu.com/widescreen/home' target='_blank' name='tj_jiankang'><img src='https://pss.bdstatic.com/static/superman/img/topnav/newjiankang-f03b804b4b.png'/><div class="s-top-more-title c-font-normal c-color-t">健康</div></a></div><div class='s-top-more-content clearfix'><a class="img-spacing" href='http://e.baidu.com/ebaidu/home?refer=887' target='_blank' name='tj_yingxiaotuiguang'><img src='https://pss.bdstatic.com/static/superman/img/topnav/yingxiaoicon-612169cc36.png'/><div class="s-top-more-title c-font-normal c-color-t">营销推广</div></a></div><div class='s-top-more-content clearfix'><a class="img-spacing" href='https://live.baidu.com/' target='_blank' name='tj_live'><img src='https://pss.bdstatic.com/static/superman/img/topnav/newzhibo-a6a0831ecd.png'/><div class="s-top-more-title c-font-normal c-color-t">直播</div></a></div><div class='s-top-more-content clearfix'><a class="img-spacing" href='http://music.taihe.com' target='_blank' name='tj_mp3'><img src='https://pss.bdstatic.com/static/superman/img/topnav/newyinyue-03ecd1e9b9.png'/><div class="s-top-more-title c-font-normal c-color-t">音乐</div></a></div><div class='s-top-more-content clearfix'><a class="img-spacing" href='https://cp.baidu.com/?sa=bdindex' target='_blank' name='tj_chengpian'><img src='https://pss.bdstatic.com/static/superman/img/topnav/chengpian-9981cd1fdb.png'/><div class="s-top-more-title c-font-normal c-color-t">橙篇</div></a></div><div class="s-top-tomore"><a class="c-color-gray2 c-font-normal" href='http://www.baidu.com/more/' target='_blank' name='tj_more'>查看全部百度产品 ></a></div></div></div></div><div id="u1" class="s-top-right s-isindex-wrap"><div class="s-weather-wrapper"><div id="s_mod_weather" class="s-mod-weather s-isindex-wrap hide-unknow-city"><div class="weather-mod"><a class="city-wather" href="//www.baidu.com/s?tn=baidutop10&amp;rsv_idx=2&amp;wd=上海天气预报" target="_blank" class="weather-mod-link"><div class="show-weather"><span class="show-city"><span class="show-city-name c-font-normal c-color-t" data-key="3800">普陀</span></span><span class="show-icon"><img class="weather-icon" src="https://dss0.bdstatic.com/5aV1bjqh_Q23odCf/static/superman/img/weather/icons/a2.png"/><span class="show-icon-temp c-font-normal c-color-t">13℃</span></span><span class="show-temp"></span></div></a><div class="show-pollution "><span class="show-airParm polution-level-10"><span class="show-polution-name">良</span></span></div></div><div class="unknown-city"><span class="unknown-icon c-icon">&#xe60f;</span><span class="unknown-setting c-font-normal c-color-t">查看天气信息</span></div></div></div><span class="s-top-right-text c-font-normal c-color-t s-top-right-new" id="s-usersetting-top" name="tj_settingicon">设置</span><a id="s-top-username" class="s-top-username s-top-right-new" data-tid="2004" href="https://www.baidu.com/my/index" target="_blank"><span class="s-top-img-wrapper"><img src="https://himg.bdimg.com/sys/portraitn/item/public.1.32197c18.0XYvCY6zrzOAiO6EuOEwag?_d=29051186"/></span><span class="user-name c-font-normal c-color-t">弹定的药</span></a><div id="s-user-name-menu" class="s-top-userset-menu c-floating-box c-font-normal"><a href="https://www.baidu.com/my/index" class="user-menu-item" target="_blank" data-tid="1000">
            个人中心
        </a><a href="http://passport.baidu.com/" class="user-menu-item" data-tid="1001" target="_blank">
            账号设置
        </a><a class="s-switch-accunt" href="javascript:;">切换账号</a><a class="quit" style="overflow:hidden" class="user-menu-item" href="javascript:;" onclick="return false;">
            退出登录
        </a></div><div id="s-user-setting-menu" class="s-top-userset-menu c-floating-box c-font-normal  hiding-feed"><div class="s-user-setting-pfmenu"></div><a class="s-set-feed hide-feed" href="javascript:;">隐藏热搜</a><a class="s-set-feed show-feed" href="javascript:;">展示热搜</a><span class="split-line"></span><a class="s-set-skin" href="javascript:;" data-tid="2001">更换皮肤</a></div><div class="guide-info guide-info-login"><i class="c-icon guide-icon">&#xe625;</i><span>新春焕新肤，设置后即可拥有！</span><i class="c-icon guide-close">&#xe610;</i></div></div><style>
.s_lg_img_gold_showre {
    display: none !important;
}
.index-logo-srcnew,
.index-logo-peak {
    display: none;
}
@media (-webkit-min-device-pixel-ratio: 2),
    (min--moz-device-pixel-ratio: 2),
    (-o-min-device-pixel-ratio: 2),
    (min-device-pixel-ratio: 2){ 
    .s_lg_img_gold_show {
        display: none !important;
    }
    .s_lg_img_gold_showre {
        display: inline !important;
    }
    .index-logo-src {
        display: none;
    }
    .index-logo-srcnew {
        display: inline;
    }
}
</style><div class='clear'></div><div
    id="head_wrapper"
    class="
        s-isindex-wrap head_wrapper s-title-img s-ps-islite
        
    "
><div id="s_fm" class="s_form s_form_login  "><div class="s_form_wrapper " id="s_form_wrapper"><div id="lg" class="s-p-top"><img id="s_lg_img" class='s_lg_img_gold_show' src="//www.baidu.com/img/PCtm_d9c8750bed0b3c7d089fa7d55720d6cf.png" width="270"  height="129" onerror="this.src='https://pss.bdstatic.com/static/superman/img/logo/bd_logo1-66368c33f8.png';this.onerror=null;" usemap="#mp" /><img id="s_lg_img_new" class='s_lg_img_gold_showre' src="//www.baidu.com/img/PCtm_d9c8750bed0b3c7d089fa7d55720d6cf.png" width="270"  height="129" onerror="this.src='https://pss.bdstatic.com/static/superman/img/logo/bd_logo1-66368c33f8.png';this.onerror=null;" usemap="#mp" /><img hidefocus="true" id="s_lg_img_aging" class='index-logo-aging-tools' src="//www.baidu.com/img/PCfb_5bf082d29588c07f842ccde3f97243ea.png" width="270" height="129" onerror="this.src='https://pss.bdstatic.com/static/superman/img/logo/bd_logo1-66368c33f8.png';this.onerror=null;" usemap="#mp"><map name="mp" id="s_mp"><area style="cursor:pointer;outline:none;" shape="rect" coords="0,0,270,129" href="//www.baidu.com/s?wd=%E7%99%BE%E5%BA%A6%E7%83%AD%E6%90%9C&amp;sa=ire_dl_gh_logo_texing&amp;rsv_dl=igh_logo_pc" target="_blank" style="outline:none;" title="点击一下，了解更多" onmousedown=&quot;return ns_c({&#39;fm&#39;:&#39;behs&#39;,&#39;tab&#39;:&#39;bdlogo&#39;})&quot;></map></div><a href="/" id="result_logo" onmousedown="return c({'fm':'tab','tab':'logo'})"><img class="index-logo-src" src="//www.baidu.com/img/flexible/logo/pc/result.png" alt="到百度首页" title="到百度首页"><img  class='index-logo-srcnew' src="//www.baidu.com/img/flexible/logo/pc/result@2.png" alt="到百度首页" title="到百度首页"><img class='index-logo-peak' src="//www.baidu.com/img/flexible/logo/pc/peak-result.png" alt="到百度首页" title="到百度首页"></a><form name="f" id="form" action="/s" class="fm "  onsubmit="javascript:F.call('ps/sug','pssubmit');"><span id="s_kw_wrap" class="bg s_ipt_wr new-pmd quickdelete-wrap"><input type="text" class="s_ipt" name="wd" id="kw" maxlength="100" autocomplete="off"><i class="c-icon quickdelete c-color-gray2">&#xe610;</i><i class="quickdelete-line"></i></span><input type="hidden" name="rsv_spt" value="1"><input type="hidden" name="rsv_iqid" value="0xfd59a6e40027a32d"><input type="hidden" name="issp" value="1"><input type="hidden" name="f" value="8"><input type="hidden" name="rsv_bp" value="0"><input type="hidden" name="rsv_idx" value="2"><input type="hidden" name="ie" value="utf-8"><input type="hidden" name="rqlang" value=""><input type="hidden" name="tn" value="baiduhome_pg" /><input type="hidden" name="ch" value=""><span class="btn_wr s_btn_wr bg" id="s_btn_wr"><input type="submit" value="百度一下" id="su" class="btn self-btn bg s_btn"></span><span class="tools"><span id="mHolder"><div id="mCon"><span>输入法</span></div><ul id="mMenu"><li><a href="javascript:;" name="ime_hw">手写</a></li><li><a href="javascript:;" name="ime_py">拼音</a></li><li class="ln"></li><li><a href="javascript:;" name="ime_cl">关闭</a></li></ul></span><span class="bd_bear_home"></span></span></form><div id="s_new_search_guide"><script type="application/json" id="new_search_guide_bub_aiSearch">
    {"count":11802824,"prefixe":"今日已解决","prefixe_url":"个问题","text":"AI搜索DeepSeek","url":"https:\/\/chat.baidu.com\/search?isShowHello=1&pd=csaitab&setype=csaitab&extParamsJson=%7B%22enter_type%22%3A%22ai_explore_home%22%7D&usedModel=%7B%22modelName%22%3A%22DeepSeek-R1%22%7D"}
</script><div class="new_search_guide_bub"><a class="new_search_guide_bub_container" href="https://chat.baidu.com/search?isShowHello=1&amp;pd=csaitab&amp;setype=csaitab&amp;extParamsJson=%7B%22enter_type%22%3A%22ai_explore_home%22%7D&amp;usedModel=%7B%22modelName%22%3A%22DeepSeek-R1%22%7D" target="_blank"><div class="new_search_guide_bub_container_left"><svg class="search_icon" width="16" height="16" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg"><path fill-rule="evenodd" clip-rule="evenodd" d="M14.5457 6.01788L14.5487 6.01063L14.5968 5.99121L14.6162 5.98339C14.6713 5.96117 14.7285 5.93807 14.7874 5.91428C14.9255 6.47658 14.9987 7.06437 14.9987 7.66927C14.9987 9.45427 14.361 11.0903 13.3009 12.362L16.2748 15.3359L15.332 16.2787L12.3581 13.3048C11.0864 14.3649 9.45037 15.0026 7.66536 15.0026C3.61528 15.0026 0.332031 11.7193 0.332031 7.66927C0.332031 3.6192 3.61528 0.335938 7.66536 0.335938C8.16155 0.335938 8.64623 0.385219 9.11477 0.479151C9.1078 0.496174 9.10092 0.512994 9.09412 0.529594L9.09382 0.530325L9.08727 0.546327C8.69388 0.709466 8.17478 0.924994 7.74949 1.12293C7.53815 1.22129 7.29247 1.34158 7.08114 1.46837C6.99088 1.52252 6.83908 1.61761 6.6845 1.74905C3.83738 2.21722 1.66536 4.68961 1.66536 7.66927C1.66536 10.983 4.35166 13.6693 7.66536 13.6693C10.8776 13.6693 13.5002 11.145 13.6579 7.97196C13.7745 7.77144 13.8862 7.54073 13.9788 7.33992C14.1716 6.92204 14.3836 6.4098 14.5454 6.01873L14.5457 6.01788Z" fill="#000311"/></svg><svg class="four_point_star" width="8" height="8" viewBox="0 0 8 8" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M4.84784 5.89358C4.51371 6.70116 4.04756 7.82786 3.90259 7.82608C3.7661 7.82441 3.33848 6.75865 3.01733 5.95821C2.87155 5.59488 2.74771 5.28622 2.68298 5.15655C2.54689 5.08864 2.21319 4.95673 1.82557 4.80351C1.02998 4.48902 0.00726687 4.08476 5.15125e-05 3.95982C-0.0089238 3.8044 1.15691 3.32091 1.97257 2.98265C2.29634 2.84837 2.56494 2.73698 2.68298 2.67808C2.744 2.55584 2.85981 2.27296 2.99826 1.93476C3.33027 1.12377 3.7925 -0.00530049 3.94166 1.87254e-05C4.07321 0.00471001 4.50063 1.06528 4.8234 1.86618C4.97169 2.23415 5.09791 2.54732 5.16318 2.67807C5.28397 2.73835 5.56022 2.85239 5.89114 2.98899C6.69833 3.32219 7.83084 3.78968 7.82607 3.93886C7.82184 4.07137 6.76391 4.49824 5.96676 4.81988C5.6029 4.96669 5.29337 5.09158 5.16318 5.15655C5.10238 5.27835 4.98647 5.5585 4.84784 5.89358Z" fill="#335BFF"/></svg></div><div class="new_search_guide_bub_container_center">
                
                    AI搜索DeepSeek
                    <span class="prefix_text">今日已解决</span><div class="new_search_guide_bub_container_center_number_counter"><span>0</span><span>0</span><span>0</span><span>0</span><span>0</span><span>0</span><span>0</span><span>0</span></div>
                    个问题
                
            </div><div class="new_search_guide_bub_container_right"><i class="cos-icon cos-icon-arrow-right"></i></div></a></div></div><div id="s_lm_wrap" class="
                    
                        s_lm_hide
                    
                    s-isindex-wrap
                    under-searchbox-tips
                "><div id="lm-new"></div></div></div></div><div id="u"><a class="toindex" href="/">百度首页</a><span class="toindex"></span><a href="javascript:;" name="tj_settingicon" class="pf">设置<i class="c-icon c-icon-triangle-down"></i></a><a href="https://www.baidu.com/my/index" id="user" class="username"><span class="s-top-img-wrapper"><img src="https://himg.bdimg.com/sys/portraitn/item/public.1.32197c18.0XYvCY6zrzOAiO6EuOEwag?_d=29051186"/></span><span class="s-top-username">弹定的药</span></a><div class="bdpfmenu"></div></div><p class="s-skin-lm s-isindex-wrap"></p></div><div id="s_wrap" class="s-isindex-wrap"><div
        id="s_main"
        class="main clearfix c-wrapper c-wrapper-l
            
            
            c-wrapper-hot-news-all
            
        "
        style="display: none;"
    ><div id="s_mancard_main" class="s-mancacrd-main"><script>
    /* eslint-disable */
    window.isNewGridCard = 1;
</script><div class="s-menu-container"><div id="s_menu_gurd" class="s-menu-gurd c-font-medium" style="display: none;"><div id="s_ctner_menus" class="s-ctner-menus s-opacity-blank8"><div class="s-menus-outer"><div id="s_menus_wrapper" class="menus-wrapper"><span id="s_menu_mine" class="s-menu-item s-menu-mine" data-id="100">
                        我的关注
                        <span class="s-menu-item-underline"></span></span><span class="s-menu-item current" data-id="99"><i class="c-icon hotsearch-title"></i><span class="s-menu-item-underline"></span></span></div><span class="s-menu-music" data-id="3"></span></div></div></div></div><div id="s-hotsearch-wrapper" class="s-isindex-wrap s-hotsearch-wrapper  s-hotsearch-wrapper-login  have-search-guide"><div class="s-hotsearch-title"><a class="hot-title" href="https://top.baidu.com/board?platform=pc&sa=pcindex_entry" target="_blank"><div class="title-text c-font-medium c-color-t" aria-label="百度热搜"><i class="c-icon hot-title-icon">&#xe687;</i><i class="c-icon arrow">&#xe613;</i></div></a><a id="hotsearch-refresh-btn" class="hot-refresh c-font-normal c-color-gray2"><i class="c-icon refresh-icon">&#xe619;</i><span class="hot-refresh-text">换一换</span></a></div><ul class="s-hotsearch-content" id="hotsearch-content-wrapper"><li class="hotsearch-item odd" data-index="0"><a class="title-content  c-link c-font-medium c-line-clamp1" href="https://www.baidu.com/s?wd=%E9%AB%98%E6%B0%B4%E5%B9%B3%E5%BC%80%E6%94%BE%E6%8B%93%E5%B1%95%E5%9B%BD%E9%99%85%E5%90%88%E4%BD%9C%E6%96%B0%E7%A9%BA%E9%97%B4&amp;sa=fyb_n_homepage&amp;rsv_dl=fyb_n_homepage&amp;from=super&amp;cl=3&amp;tn=baidutop10&amp;fr=top1000&amp;rsv_idx=2&amp;hisfilter=1" target="_blank" ><div class="title-content-noindex" style="display: none;"></div><i class="c-icon title-content-top-icon c-color-red c-gap-right-small" style="display: ;">&#xe62e;</i><span class="title-content-index c-index-single c-index-single-hot0" style="display: none;">0</span><span class="title-content-title">高水平开放拓展国际合作新空间</span></a><span class="title-content-mark ie-vertical c-text c-gap-left-small "></span></li><li class="hotsearch-item even" data-index="5"><a class="title-content  c-link c-font-medium c-line-clamp1" href="https://www.baidu.com/s?wd=%E4%BA%8E%E6%AD%A3%E6%9B%9D%E6%9F%90%E7%94%B7%E6%BC%94%E5%91%98%E5%B0%86%E5%A5%B3%E4%B8%BB%E6%88%8F%E5%87%A0%E4%B9%8E%E5%88%A0%E6%B2%A1%E4%BA%86&amp;sa=fyb_n_homepage&amp;rsv_dl=fyb_n_homepage&amp;from=super&amp;cl=3&amp;tn=baidutop10&amp;fr=top1000&amp;rsv_idx=2&amp;hisfilter=1" target="_blank" ><div class="title-content-noindex" style="display: none;"></div><i class="c-icon title-content-top-icon c-color-red c-gap-right-small" style="display: none;">&#xe62e;</i><span class="title-content-index c-index-single c-index-single-hot5" style="display: ;">5</span><span class="title-content-title">于正曝某男演员将女主戏几乎删没了</span></a><span class="title-content-mark ie-vertical c-text c-gap-left-small "></span></li><li class="hotsearch-item odd" data-index="1"><a class="title-content tag-width c-link c-font-medium c-line-clamp1" href="https://www.baidu.com/s?wd=%E5%A4%A7%E6%A1%A5%E5%9E%AE%E5%A1%8C%E8%87%B462%E4%BA%BA%E6%AD%BB%E4%BA%A1%E5%A4%B1%E8%B8%AA+43%E4%BA%BA%E8%A2%AB%E9%97%AE%E8%B4%A3&amp;sa=fyb_n_homepage&amp;rsv_dl=fyb_n_homepage&amp;from=super&amp;cl=3&amp;tn=baidutop10&amp;fr=top1000&amp;rsv_idx=2&amp;hisfilter=1" target="_blank" ><div class="title-content-noindex" style="display: none;"></div><i class="c-icon title-content-top-icon c-color-red c-gap-right-small" style="display: none;">&#xe62e;</i><span class="title-content-index c-index-single c-index-single-hot1" style="display: ;">1</span><span class="title-content-title">大桥垮塌致62人死亡失踪 43人被问责</span></a><span class="title-content-mark ie-vertical c-text c-gap-left-small c-text-fei">沸</span></li><li class="hotsearch-item even" data-index="6"><a class="title-content  c-link c-font-medium c-line-clamp1" href="https://www.baidu.com/s?wd=%E2%80%9C%E8%83%96%E9%83%BD%E6%9D%A5%E2%80%9D%E6%88%90%E7%AB%8B+%E8%83%96%E4%B8%9C%E6%9D%A5%E5%9B%9E%E5%BA%94&amp;sa=fyb_n_homepage&amp;rsv_dl=fyb_n_homepage&amp;from=super&amp;cl=3&amp;tn=baidutop10&amp;fr=top1000&amp;rsv_idx=2&amp;hisfilter=1" target="_blank" ><div class="title-content-noindex" style="display: none;"></div><i class="c-icon title-content-top-icon c-color-red c-gap-right-small" style="display: none;">&#xe62e;</i><span class="title-content-index c-index-single c-index-single-hot6" style="display: ;">6</span><span class="title-content-title">“胖都来”成立 胖东来回应</span></a><span class="title-content-mark ie-vertical c-text c-gap-left-small "></span></li><li class="hotsearch-item odd" data-index="2"><a class="title-content tag-width c-link c-font-medium c-line-clamp1" href="https://www.baidu.com/s?wd=DeepSeek%E5%88%9B%E5%A7%8B%E4%BA%BA%E9%A6%96%E7%99%BB%E5%85%A8%E7%90%83%E5%AF%8C%E8%B1%AA%E6%A6%9C&amp;sa=fyb_n_homepage&amp;rsv_dl=fyb_n_homepage&amp;from=super&amp;cl=3&amp;tn=baidutop10&amp;fr=top1000&amp;rsv_idx=2&amp;hisfilter=1" target="_blank" ><div class="title-content-noindex" style="display: none;"></div><i class="c-icon title-content-top-icon c-color-red c-gap-right-small" style="display: none;">&#xe62e;</i><span class="title-content-index c-index-single c-index-single-hot2" style="display: ;">2</span><span class="title-content-title">DeepSeek创始人首登全球富豪榜</span></a><span class="title-content-mark ie-vertical c-text c-gap-left-small c-text-hot">热</span></li><li class="hotsearch-item even" data-index="7"><a class="title-content  c-link c-font-medium c-line-clamp1" href="https://www.baidu.com/s?wd=%E4%B8%80%E6%B1%BD%E5%A5%94%E8%85%BE%E6%82%A6%E6%84%8F%E6%9D%A5%E4%BA%86&amp;sa=fyb_n_homepage&amp;rsv_dl=fyb_n_homepage&amp;from=super&amp;cl=3&amp;tn=baidutop10&amp;fr=top1000&amp;rsv_idx=2&amp;hisfilter=1" target="_blank" ><div class="title-content-noindex" style="display: none;"></div><i class="c-icon title-content-top-icon c-color-red c-gap-right-small" style="display: none;">&#xe62e;</i><span class="title-content-index c-index-single c-index-single-hot7" style="display: ;">7</span><span class="title-content-title">一汽奔腾悦意来了</span></a><span class="title-content-mark ie-vertical c-text c-gap-left-small "></span></li><li class="hotsearch-item odd" data-index="3"><a class="title-content  c-link c-font-medium c-line-clamp1" href="https://www.baidu.com/s?wd=%E6%96%B0%E4%B8%80%E6%89%B9%E9%A3%9F%E5%93%81%E5%AE%89%E5%85%A8%E5%9B%BD%E5%AE%B6%E6%A0%87%E5%87%86%E6%9D%A5%E4%BA%86&amp;sa=fyb_n_homepage&amp;rsv_dl=fyb_n_homepage&amp;from=super&amp;cl=3&amp;tn=baidutop10&amp;fr=top1000&amp;rsv_idx=2&amp;hisfilter=1" target="_blank" ><div class="title-content-noindex" style="display: none;"></div><i class="c-icon title-content-top-icon c-color-red c-gap-right-small" style="display: none;">&#xe62e;</i><span class="title-content-index c-index-single c-index-single-hot3" style="display: ;">3</span><span class="title-content-title">新一批食品安全国家标准来了</span></a><span class="title-content-mark ie-vertical c-text c-gap-left-small "></span></li><li class="hotsearch-item even" data-index="8"><a class="title-content  c-link c-font-medium c-line-clamp1" href="https://www.baidu.com/s?wd=%E5%A4%96%E4%BA%A4%E9%83%A8%E5%9B%9E%E5%BA%94%E6%9D%8E%E5%98%89%E8%AF%9A%E9%9B%86%E5%9B%A2%E5%87%BA%E5%94%AE%E6%B5%B7%E5%A4%96%E6%B8%AF%E5%8F%A3&amp;sa=fyb_n_homepage&amp;rsv_dl=fyb_n_homepage&amp;from=super&amp;cl=3&amp;tn=baidutop10&amp;fr=top1000&amp;rsv_idx=2&amp;hisfilter=1" target="_blank" ><div class="title-content-noindex" style="display: none;"></div><i class="c-icon title-content-top-icon c-color-red c-gap-right-small" style="display: none;">&#xe62e;</i><span class="title-content-index c-index-single c-index-single-hot8" style="display: ;">8</span><span class="title-content-title">外交部回应李嘉诚集团出售海外港口</span></a><span class="title-content-mark ie-vertical c-text c-gap-left-small "></span></li><li class="hotsearch-item odd" data-index="4"><a class="title-content tag-width c-link c-font-medium c-line-clamp1" href="https://www.baidu.com/s?wd=%E9%A6%99%E9%A3%98%E9%A3%98%E5%A5%B6%E8%8C%B6%E6%B6%A8%E4%BB%B7%E8%87%B316%E5%85%83%E4%B8%80%E6%9D%AF&amp;sa=fyb_n_homepage&amp;rsv_dl=fyb_n_homepage&amp;from=super&amp;cl=3&amp;tn=baidutop10&amp;fr=top1000&amp;rsv_idx=2&amp;hisfilter=1" target="_blank" ><div class="title-content-noindex" style="display: none;"></div><i class="c-icon title-content-top-icon c-color-red c-gap-right-small" style="display: none;">&#xe62e;</i><span class="title-content-index c-index-single c-index-single-hot4" style="display: ;">4</span><span class="title-content-title">香飘飘奶茶涨价至16元一杯</span></a><span class="title-content-mark ie-vertical c-text c-gap-left-small c-text-hot">热</span></li><li class="hotsearch-item even" data-index="9"><a class="title-content  c-link c-font-medium c-line-clamp1" href="https://www.baidu.com/s?wd=%E8%B5%B5%E9%9C%B2%E6%80%9D%E6%89%BF%E8%AE%A4%E8%87%AA%E5%B7%B1%E5%AF%B9%E7%94%9F%E5%91%BD%E6%B2%A1%E6%95%AC%E7%95%8F&amp;sa=fyb_n_homepage&amp;rsv_dl=fyb_n_homepage&amp;from=super&amp;cl=3&amp;tn=baidutop10&amp;fr=top1000&amp;rsv_idx=2&amp;hisfilter=1" target="_blank" ><div class="title-content-noindex" style="display: none;"></div><i class="c-icon title-content-top-icon c-color-red c-gap-right-small" style="display: none;">&#xe62e;</i><span class="title-content-index c-index-single c-index-single-hot9" style="display: ;">9</span><span class="title-content-title">赵露思承认自己对生命没敬畏</span></a><span class="title-content-mark ie-vertical c-text c-gap-left-small "></span></li></ul></div><textarea id="hotsearch_data" style="display:none;">{
    "hotsearch": [
        
            
            {
                "card_title": "高水平开放拓展国际合作新空间",
                "linkurl": "https%3A/%2Fwww.baidu.com%2Fs%3Fwd%3D%25E9%25AB%2598%25E6%25B0%25B4%25E5%25B9%25B3%25E5%25BC%2580%25E6%2594%25BE%25E6%258B%2593%25E5%25B1%2595%25E5%259B%25BD%25E9%2599%2585%25E5%2590%2588%25E4%25BD%259C%25E6%2596%25B0%25E7%25A9%25BA%25E9%2597%25B4%26sa%3Dfyb_n_homepage%26rsv_dl%3Dfyb_n_homepage%26from%3Dsuper%26cl%3D3%26tn%3Dbaidutop10%26fr%3Dtop1000%26rsv_idx%3D2%26hisfilter%3D1",
                "views": "",
                "isViewed": "",
                "isNew": "",
                "heat_score": "7990475",
                "hotTags": "0",
                "pre_tag": "",
                "index": "0"
            }
        
            ,
            {
                "card_title": "大桥垮塌致62人死亡失踪 43人被问责",
                "linkurl": "https%3A/%2Fwww.baidu.com%2Fs%3Fwd%3D%25E5%25A4%25A7%25E6%25A1%25A5%25E5%259E%25AE%25E5%25A1%258C%25E8%2587%25B462%25E4%25BA%25BA%25E6%25AD%25BB%25E4%25BA%25A1%25E5%25A4%25B1%25E8%25B8%25AA%2B43%25E4%25BA%25BA%25E8%25A2%25AB%25E9%2597%25AE%25E8%25B4%25A3%26sa%3Dfyb_n_homepage%26rsv_dl%3Dfyb_n_homepage%26from%3Dsuper%26cl%3D3%26tn%3Dbaidutop10%26fr%3Dtop1000%26rsv_idx%3D2%26hisfilter%3D1",
                "views": "",
                "isViewed": "",
                "isNew": "",
                "heat_score": "7913081",
                "hotTags": "4",
                "pre_tag": "",
                "index": "1"
            }
        
            ,
            {
                "card_title": "DeepSeek创始人首登全球富豪榜",
                "linkurl": "https%3A/%2Fwww.baidu.com%2Fs%3Fwd%3DDeepSeek%25E5%2588%259B%25E5%25A7%258B%25E4%25BA%25BA%25E9%25A6%2596%25E7%2599%25BB%25E5%2585%25A8%25E7%2590%2583%25E5%25AF%258C%25E8%25B1%25AA%25E6%25A6%259C%26sa%3Dfyb_n_homepage%26rsv_dl%3Dfyb_n_homepage%26from%3Dsuper%26cl%3D3%26tn%3Dbaidutop10%26fr%3Dtop1000%26rsv_idx%3D2%26hisfilter%3D1",
                "views": "",
                "isViewed": "",
                "isNew": "",
                "heat_score": "7806164",
                "hotTags": "3",
                "pre_tag": "",
                "index": "2"
            }
        
            ,
            {
                "card_title": "新一批食品安全国家标准来了",
                "linkurl": "https%3A/%2Fwww.baidu.com%2Fs%3Fwd%3D%25E6%2596%25B0%25E4%25B8%2580%25E6%2589%25B9%25E9%25A3%259F%25E5%2593%2581%25E5%25AE%2589%25E5%2585%25A8%25E5%259B%25BD%25E5%25AE%25B6%25E6%25A0%2587%25E5%2587%2586%25E6%259D%25A5%25E4%25BA%2586%26sa%3Dfyb_n_homepage%26rsv_dl%3Dfyb_n_homepage%26from%3Dsuper%26cl%3D3%26tn%3Dbaidutop10%26fr%3Dtop1000%26rsv_idx%3D2%26hisfilter%3D1",
                "views": "",
                "isViewed": "",
                "isNew": "",
                "heat_score": "7761903",
                "hotTags": "0",
                "pre_tag": "",
                "index": "3"
            }
        
            ,
            {
                "card_title": "香飘飘奶茶涨价至16元一杯",
                "linkurl": "https%3A/%2Fwww.baidu.com%2Fs%3Fwd%3D%25E9%25A6%2599%25E9%25A3%2598%25E9%25A3%2598%25E5%25A5%25B6%25E8%258C%25B6%25E6%25B6%25A8%25E4%25BB%25B7%25E8%2587%25B316%25E5%2585%2583%25E4%25B8%2580%25E6%259D%25AF%26sa%3Dfyb_n_homepage%26rsv_dl%3Dfyb_n_homepage%26from%3Dsuper%26cl%3D3%26tn%3Dbaidutop10%26fr%3Dtop1000%26rsv_idx%3D2%26hisfilter%3D1",
                "views": "",
                "isViewed": "",
                "isNew": "",
                "heat_score": "7633546",
                "hotTags": "3",
                "pre_tag": "",
                "index": "4"
            }
        
            ,
            {
                "card_title": "于正曝某男演员将女主戏几乎删没了",
                "linkurl": "https%3A/%2Fwww.baidu.com%2Fs%3Fwd%3D%25E4%25BA%258E%25E6%25AD%25A3%25E6%259B%259D%25E6%259F%2590%25E7%2594%25B7%25E6%25BC%2594%25E5%2591%2598%25E5%25B0%2586%25E5%25A5%25B3%25E4%25B8%25BB%25E6%2588%258F%25E5%2587%25A0%25E4%25B9%258E%25E5%2588%25A0%25E6%25B2%25A1%25E4%25BA%2586%26sa%3Dfyb_n_homepage%26rsv_dl%3Dfyb_n_homepage%26from%3Dsuper%26cl%3D3%26tn%3Dbaidutop10%26fr%3Dtop1000%26rsv_idx%3D2%26hisfilter%3D1",
                "views": "",
                "isViewed": "",
                "isNew": "",
                "heat_score": "7514493",
                "hotTags": "0",
                "pre_tag": "",
                "index": "5"
            }
        
            ,
            {
                "card_title": "“胖都来”成立 胖东来回应",
                "linkurl": "https%3A/%2Fwww.baidu.com%2Fs%3Fwd%3D%25E2%2580%259C%25E8%2583%2596%25E9%2583%25BD%25E6%259D%25A5%25E2%2580%259D%25E6%2588%2590%25E7%25AB%258B%2B%25E8%2583%2596%25E4%25B8%259C%25E6%259D%25A5%25E5%259B%259E%25E5%25BA%2594%26sa%3Dfyb_n_homepage%26rsv_dl%3Dfyb_n_homepage%26from%3Dsuper%26cl%3D3%26tn%3Dbaidutop10%26fr%3Dtop1000%26rsv_idx%3D2%26hisfilter%3D1",
                "views": "",
                "isViewed": "",
                "isNew": "",
                "heat_score": "7407188",
                "hotTags": "0",
                "pre_tag": "",
                "index": "6"
            }
        
            ,
            {
                "card_title": "一汽奔腾悦意来了",
                "linkurl": "https%3A/%2Fwww.baidu.com%2Fs%3Fwd%3D%25E4%25B8%2580%25E6%25B1%25BD%25E5%25A5%2594%25E8%2585%25BE%25E6%2582%25A6%25E6%2584%258F%25E6%259D%25A5%25E4%25BA%2586%26sa%3Dfyb_n_homepage%26rsv_dl%3Dfyb_n_homepage%26from%3Dsuper%26cl%3D3%26tn%3Dbaidutop10%26fr%3Dtop1000%26rsv_idx%3D2%26hisfilter%3D1",
                "views": "",
                "isViewed": "",
                "isNew": "",
                "heat_score": "7358287",
                "hotTags": "0",
                "pre_tag": "",
                "index": "7"
            }
        
            ,
            {
                "card_title": "外交部回应李嘉诚集团出售海外港口",
                "linkurl": "https%3A/%2Fwww.baidu.com%2Fs%3Fwd%3D%25E5%25A4%2596%25E4%25BA%25A4%25E9%2583%25A8%25E5%259B%259E%25E5%25BA%2594%25E6%259D%258E%25E5%2598%2589%25E8%25AF%259A%25E9%259B%2586%25E5%259B%25A2%25E5%2587%25BA%25E5%2594%25AE%25E6%25B5%25B7%25E5%25A4%2596%25E6%25B8%25AF%25E5%258F%25A3%26sa%3Dfyb_n_homepage%26rsv_dl%3Dfyb_n_homepage%26from%3Dsuper%26cl%3D3%26tn%3Dbaidutop10%26fr%3Dtop1000%26rsv_idx%3D2%26hisfilter%3D1",
                "views": "",
                "isViewed": "",
                "isNew": "",
                "heat_score": "7232578",
                "hotTags": "0",
                "pre_tag": "",
                "index": "8"
            }
        
            ,
            {
                "card_title": "赵露思承认自己对生命没敬畏",
                "linkurl": "https%3A/%2Fwww.baidu.com%2Fs%3Fwd%3D%25E8%25B5%25B5%25E9%259C%25B2%25E6%2580%259D%25E6%2589%25BF%25E8%25AE%25A4%25E8%2587%25AA%25E5%25B7%25B1%25E5%25AF%25B9%25E7%2594%259F%25E5%2591%25BD%25E6%25B2%25A1%25E6%2595%25AC%25E7%2595%258F%26sa%3Dfyb_n_homepage%26rsv_dl%3Dfyb_n_homepage%26from%3Dsuper%26cl%3D3%26tn%3Dbaidutop10%26fr%3Dtop1000%26rsv_idx%3D2%26hisfilter%3D1",
                "views": "",
                "isViewed": "",
                "isNew": "",
                "heat_score": "7164090",
                "hotTags": "0",
                "pre_tag": "",
                "index": "9"
            }
        
            ,
            {
                "card_title": "爸爸去世女儿被判和4个姑姑平分遗产",
                "linkurl": "https%3A/%2Fwww.baidu.com%2Fs%3Fwd%3D%25E7%2588%25B8%25E7%2588%25B8%25E5%258E%25BB%25E4%25B8%2596%25E5%25A5%25B3%25E5%2584%25BF%25E8%25A2%25AB%25E5%2588%25A4%25E5%2592%258C4%25E4%25B8%25AA%25E5%25A7%2591%25E5%25A7%2591%25E5%25B9%25B3%25E5%2588%2586%25E9%2581%2597%25E4%25BA%25A7%26sa%3Dfyb_n_homepage%26rsv_dl%3Dfyb_n_homepage%26from%3Dsuper%26cl%3D3%26tn%3Dbaidutop10%26fr%3Dtop1000%26rsv_idx%3D2%26hisfilter%3D1",
                "views": "",
                "isViewed": "",
                "isNew": "",
                "heat_score": "7057881",
                "hotTags": "3",
                "pre_tag": "",
                "index": "10"
            }
        
            ,
            {
                "card_title": "1人造谣“三亚变空城”被处罚",
                "linkurl": "https%3A/%2Fwww.baidu.com%2Fs%3Fwd%3D1%25E4%25BA%25BA%25E9%2580%25A0%25E8%25B0%25A3%25E2%2580%259C%25E4%25B8%2589%25E4%25BA%259A%25E5%258F%2598%25E7%25A9%25BA%25E5%259F%258E%25E2%2580%259D%25E8%25A2%25AB%25E5%25A4%2584%25E7%25BD%259A%26sa%3Dfyb_n_homepage%26rsv_dl%3Dfyb_n_homepage%26from%3Dsuper%26cl%3D3%26tn%3Dbaidutop10%26fr%3Dtop1000%26rsv_idx%3D2%26hisfilter%3D1",
                "views": "",
                "isViewed": "",
                "isNew": "",
                "heat_score": "6981671",
                "hotTags": "0",
                "pre_tag": "",
                "index": "11"
            }
        
            ,
            {
                "card_title": "儿子给腰突父亲买40小时硬座票",
                "linkurl": "https%3A/%2Fwww.baidu.com%2Fs%3Fwd%3D%25E5%2584%25BF%25E5%25AD%2590%25E7%25BB%2599%25E8%2585%25B0%25E7%25AA%2581%25E7%2588%25B6%25E4%25BA%25B2%25E4%25B9%25B040%25E5%25B0%258F%25E6%2597%25B6%25E7%25A1%25AC%25E5%25BA%25A7%25E7%25A5%25A8%26sa%3Dfyb_n_homepage%26rsv_dl%3Dfyb_n_homepage%26from%3Dsuper%26cl%3D3%26tn%3Dbaidutop10%26fr%3Dtop1000%26rsv_idx%3D2%26hisfilter%3D1",
                "views": "",
                "isViewed": "",
                "isNew": "",
                "heat_score": "6806746",
                "hotTags": "0",
                "pre_tag": "",
                "index": "12"
            }
        
            ,
            {
                "card_title": "李在明被判无罪后跪地慰问山火灾民",
                "linkurl": "https%3A/%2Fwww.baidu.com%2Fs%3Fwd%3D%25E6%259D%258E%25E5%259C%25A8%25E6%2598%258E%25E8%25A2%25AB%25E5%2588%25A4%25E6%2597%25A0%25E7%25BD%25AA%25E5%2590%258E%25E8%25B7%25AA%25E5%259C%25B0%25E6%2585%25B0%25E9%2597%25AE%25E5%25B1%25B1%25E7%2581%25AB%25E7%2581%25BE%25E6%25B0%2591%26sa%3Dfyb_n_homepage%26rsv_dl%3Dfyb_n_homepage%26from%3Dsuper%26cl%3D3%26tn%3Dbaidutop10%26fr%3Dtop1000%26rsv_idx%3D2%26hisfilter%3D1",
                "views": "",
                "isViewed": "",
                "isNew": "",
                "heat_score": "6750093",
                "hotTags": "0",
                "pre_tag": "",
                "index": "13"
            }
        
            ,
            {
                "card_title": "凤凰传奇曾毅勇闯电影圈",
                "linkurl": "https%3A/%2Fwww.baidu.com%2Fs%3Fwd%3D%25E5%2587%25A4%25E5%2587%25B0%25E4%25BC%25A0%25E5%25A5%2587%25E6%259B%25BE%25E6%25AF%2585%25E5%258B%2587%25E9%2597%25AF%25E7%2594%25B5%25E5%25BD%25B1%25E5%259C%2588%26sa%3Dfyb_n_homepage%26rsv_dl%3Dfyb_n_homepage%26from%3Dsuper%26cl%3D3%26tn%3Dbaidutop10%26fr%3Dtop1000%26rsv_idx%3D2%26hisfilter%3D1",
                "views": "",
                "isViewed": "",
                "isNew": "",
                "heat_score": "6685566",
                "hotTags": "0",
                "pre_tag": "",
                "index": "14"
            }
        
            ,
            {
                "card_title": "杭州女童坠亡案雇主被保姆女儿起诉",
                "linkurl": "https%3A/%2Fwww.baidu.com%2Fs%3Fwd%3D%25E6%259D%25AD%25E5%25B7%259E%25E5%25A5%25B3%25E7%25AB%25A5%25E5%259D%25A0%25E4%25BA%25A1%25E6%25A1%2588%25E9%259B%2587%25E4%25B8%25BB%25E8%25A2%25AB%25E4%25BF%259D%25E5%25A7%2586%25E5%25A5%25B3%25E5%2584%25BF%25E8%25B5%25B7%25E8%25AF%2589%26sa%3Dfyb_n_homepage%26rsv_dl%3Dfyb_n_homepage%26from%3Dsuper%26cl%3D3%26tn%3Dbaidutop10%26fr%3Dtop1000%26rsv_idx%3D2%26hisfilter%3D1",
                "views": "",
                "isViewed": "",
                "isNew": "",
                "heat_score": "6562148",
                "hotTags": "0",
                "pre_tag": "",
                "index": "15"
            }
        
            ,
            {
                "card_title": "你以为的毒贩vs你不知道的伪装",
                "linkurl": "https%3A/%2Fwww.baidu.com%2Fs%3Fwd%3D%25E4%25BD%25A0%25E4%25BB%25A5%25E4%25B8%25BA%25E7%259A%2584%25E6%25AF%2592%25E8%25B4%25A9vs%25E4%25BD%25A0%25E4%25B8%258D%25E7%259F%25A5%25E9%2581%2593%25E7%259A%2584%25E4%25BC%25AA%25E8%25A3%2585%26sa%3Dfyb_n_homepage%26rsv_dl%3Dfyb_n_homepage%26from%3Dsuper%26cl%3D3%26tn%3Dbaidutop10%26fr%3Dtop1000%26rsv_idx%3D2%26hisfilter%3D1",
                "views": "",
                "isViewed": "",
                "isNew": "",
                "heat_score": "6496673",
                "hotTags": "0",
                "pre_tag": "",
                "index": "16"
            }
        
            ,
            {
                "card_title": "网红胡楚靓确诊多囊卵巢综合症",
                "linkurl": "https%3A/%2Fwww.baidu.com%2Fs%3Fwd%3D%25E7%25BD%2591%25E7%25BA%25A2%25E8%2583%25A1%25E6%25A5%259A%25E9%259D%2593%25E7%25A1%25AE%25E8%25AF%258A%25E5%25A4%259A%25E5%259B%258A%25E5%258D%25B5%25E5%25B7%25A2%25E7%25BB%25BC%25E5%2590%2588%25E7%2597%2587%26sa%3Dfyb_n_homepage%26rsv_dl%3Dfyb_n_homepage%26from%3Dsuper%26cl%3D3%26tn%3Dbaidutop10%26fr%3Dtop1000%26rsv_idx%3D2%26hisfilter%3D1",
                "views": "",
                "isViewed": "",
                "isNew": "",
                "heat_score": "6398232",
                "hotTags": "0",
                "pre_tag": "",
                "index": "17"
            }
        
            ,
            {
                "card_title": "赚70万司机给妻子转200自责太少",
                "linkurl": "https%3A/%2Fwww.baidu.com%2Fs%3Fwd%3D%25E8%25B5%259A70%25E4%25B8%2587%25E5%258F%25B8%25E6%259C%25BA%25E7%25BB%2599%25E5%25A6%25BB%25E5%25AD%2590%25E8%25BD%25AC200%25E8%2587%25AA%25E8%25B4%25A3%25E5%25A4%25AA%25E5%25B0%2591%26sa%3Dfyb_n_homepage%26rsv_dl%3Dfyb_n_homepage%26from%3Dsuper%26cl%3D3%26tn%3Dbaidutop10%26fr%3Dtop1000%26rsv_idx%3D2%26hisfilter%3D1",
                "views": "",
                "isViewed": "",
                "isNew": "",
                "heat_score": "6288829",
                "hotTags": "0",
                "pre_tag": "",
                "index": "18"
            }
        
            ,
            {
                "card_title": "男子在工地宿舍睡觉时被捅身亡",
                "linkurl": "https%3A/%2Fwww.baidu.com%2Fs%3Fwd%3D%25E7%2594%25B7%25E5%25AD%2590%25E5%259C%25A8%25E5%25B7%25A5%25E5%259C%25B0%25E5%25AE%25BF%25E8%2588%258D%25E7%259D%25A1%25E8%25A7%2589%25E6%2597%25B6%25E8%25A2%25AB%25E6%258D%2585%25E8%25BA%25AB%25E4%25BA%25A1%26sa%3Dfyb_n_homepage%26rsv_dl%3Dfyb_n_homepage%26from%3Dsuper%26cl%3D3%26tn%3Dbaidutop10%26fr%3Dtop1000%26rsv_idx%3D2%26hisfilter%3D1",
                "views": "",
                "isViewed": "",
                "isNew": "",
                "heat_score": "6184725",
                "hotTags": "0",
                "pre_tag": "",
                "index": "19"
            }
        
            ,
            {
                "card_title": "财政局副科长被举报出轨 纪委回应",
                "linkurl": "https%3A/%2Fwww.baidu.com%2Fs%3Fwd%3D%25E8%25B4%25A2%25E6%2594%25BF%25E5%25B1%2580%25E5%2589%25AF%25E7%25A7%2591%25E9%2595%25BF%25E8%25A2%25AB%25E4%25B8%25BE%25E6%258A%25A5%25E5%2587%25BA%25E8%25BD%25A8%2B%25E7%25BA%25AA%25E5%25A7%2594%25E5%259B%259E%25E5%25BA%2594%26sa%3Dfyb_n_homepage%26rsv_dl%3Dfyb_n_homepage%26from%3Dsuper%26cl%3D3%26tn%3Dbaidutop10%26fr%3Dtop1000%26rsv_idx%3D2%26hisfilter%3D1",
                "views": "",
                "isViewed": "",
                "isNew": "",
                "heat_score": "6066608",
                "hotTags": "0",
                "pre_tag": "",
                "index": "20"
            }
        
            ,
            {
                "card_title": "李晟被扒出曾背刺琼瑶黑历史",
                "linkurl": "https%3A/%2Fwww.baidu.com%2Fs%3Fwd%3D%25E6%259D%258E%25E6%2599%259F%25E8%25A2%25AB%25E6%2589%2592%25E5%2587%25BA%25E6%259B%25BE%25E8%2583%258C%25E5%2588%25BA%25E7%2590%25BC%25E7%2591%25B6%25E9%25BB%2591%25E5%258E%2586%25E5%258F%25B2%26sa%3Dfyb_n_homepage%26rsv_dl%3Dfyb_n_homepage%26from%3Dsuper%26cl%3D3%26tn%3Dbaidutop10%26fr%3Dtop1000%26rsv_idx%3D2%26hisfilter%3D1",
                "views": "",
                "isViewed": "",
                "isNew": "",
                "heat_score": "5999232",
                "hotTags": "0",
                "pre_tag": "",
                "index": "21"
            }
        
            ,
            {
                "card_title": "贝莱德“吞下”李嘉诚43个港口背后",
                "linkurl": "https%3A/%2Fwww.baidu.com%2Fs%3Fwd%3D%25E8%25B4%259D%25E8%258E%25B1%25E5%25BE%25B7%25E2%2580%259C%25E5%2590%259E%25E4%25B8%258B%25E2%2580%259D%25E6%259D%258E%25E5%2598%2589%25E8%25AF%259A43%25E4%25B8%25AA%25E6%25B8%25AF%25E5%258F%25A3%25E8%2583%258C%25E5%2590%258E%26sa%3Dfyb_n_homepage%26rsv_dl%3Dfyb_n_homepage%26from%3Dsuper%26cl%3D3%26tn%3Dbaidutop10%26fr%3Dtop1000%26rsv_idx%3D2%26hisfilter%3D1",
                "views": "",
                "isViewed": "",
                "isNew": "",
                "heat_score": "5862817",
                "hotTags": "0",
                "pre_tag": "",
                "index": "22"
            }
        
            ,
            {
                "card_title": "媒体：加新总理是特朗普强硬对手",
                "linkurl": "https%3A/%2Fwww.baidu.com%2Fs%3Fwd%3D%25E5%25AA%2592%25E4%25BD%2593%25EF%25BC%259A%25E5%258A%25A0%25E6%2596%25B0%25E6%2580%25BB%25E7%2590%2586%25E6%2598%25AF%25E7%2589%25B9%25E6%259C%2597%25E6%2599%25AE%25E5%25BC%25BA%25E7%25A1%25AC%25E5%25AF%25B9%25E6%2589%258B%26sa%3Dfyb_n_homepage%26rsv_dl%3Dfyb_n_homepage%26from%3Dsuper%26cl%3D3%26tn%3Dbaidutop10%26fr%3Dtop1000%26rsv_idx%3D2%26hisfilter%3D1",
                "views": "",
                "isViewed": "",
                "isNew": "",
                "heat_score": "5784810",
                "hotTags": "0",
                "pre_tag": "",
                "index": "23"
            }
        
            ,
            {
                "card_title": "国防部：全时待战、随时能战、战之必胜",
                "linkurl": "https%3A/%2Fwww.baidu.com%2Fs%3Fwd%3D%25E5%259B%25BD%25E9%2598%25B2%25E9%2583%25A8%25EF%25BC%259A%25E5%2585%25A8%25E6%2597%25B6%25E5%25BE%2585%25E6%2588%2598%25E3%2580%2581%25E9%259A%258F%25E6%2597%25B6%25E8%2583%25BD%25E6%2588%2598%25E3%2580%2581%25E6%2588%2598%25E4%25B9%258B%25E5%25BF%2585%25E8%2583%259C%26sa%3Dfyb_n_homepage%26rsv_dl%3Dfyb_n_homepage%26from%3Dsuper%26cl%3D3%26tn%3Dbaidutop10%26fr%3Dtop1000%26rsv_idx%3D2%26hisfilter%3D1",
                "views": "",
                "isViewed": "",
                "isNew": "",
                "heat_score": "5667340",
                "hotTags": "0",
                "pre_tag": "",
                "index": "24"
            }
        
            ,
            {
                "card_title": "甲亢哥和大妈合影被美颜拍成外星人",
                "linkurl": "https%3A/%2Fwww.baidu.com%2Fs%3Fwd%3D%25E7%2594%25B2%25E4%25BA%25A2%25E5%2593%25A5%25E5%2592%258C%25E5%25A4%25A7%25E5%25A6%2588%25E5%2590%2588%25E5%25BD%25B1%25E8%25A2%25AB%25E7%25BE%258E%25E9%25A2%259C%25E6%258B%258D%25E6%2588%2590%25E5%25A4%2596%25E6%2598%259F%25E4%25BA%25BA%26sa%3Dfyb_n_homepage%26rsv_dl%3Dfyb_n_homepage%26from%3Dsuper%26cl%3D3%26tn%3Dbaidutop10%26fr%3Dtop1000%26rsv_idx%3D2%26hisfilter%3D1",
                "views": "",
                "isViewed": "",
                "isNew": "",
                "heat_score": "5547364",
                "hotTags": "3",
                "pre_tag": "",
                "index": "25"
            }
        
            ,
            {
                "card_title": "雷军富豪榜全球排名上升91位",
                "linkurl": "https%3A/%2Fwww.baidu.com%2Fs%3Fwd%3D%25E9%259B%25B7%25E5%2586%259B%25E5%25AF%258C%25E8%25B1%25AA%25E6%25A6%259C%25E5%2585%25A8%25E7%2590%2583%25E6%258E%2592%25E5%2590%258D%25E4%25B8%258A%25E5%258D%258791%25E4%25BD%258D%26sa%3Dfyb_n_homepage%26rsv_dl%3Dfyb_n_homepage%26from%3Dsuper%26cl%3D3%26tn%3Dbaidutop10%26fr%3Dtop1000%26rsv_idx%3D2%26hisfilter%3D1",
                "views": "",
                "isViewed": "",
                "isNew": "",
                "heat_score": "5463336",
                "hotTags": "0",
                "pre_tag": "",
                "index": "26"
            }
        
            ,
            {
                "card_title": "42岁黄圣依亮相跑马拉松",
                "linkurl": "https%3A/%2Fwww.baidu.com%2Fs%3Fwd%3D42%25E5%25B2%2581%25E9%25BB%2584%25E5%259C%25A3%25E4%25BE%259D%25E4%25BA%25AE%25E7%259B%25B8%25E8%25B7%2591%25E9%25A9%25AC%25E6%258B%2589%25E6%259D%25BE%26sa%3Dfyb_n_homepage%26rsv_dl%3Dfyb_n_homepage%26from%3Dsuper%26cl%3D3%26tn%3Dbaidutop10%26fr%3Dtop1000%26rsv_idx%3D2%26hisfilter%3D1",
                "views": "",
                "isViewed": "",
                "isNew": "",
                "heat_score": "5354622",
                "hotTags": "0",
                "pre_tag": "",
                "index": "27"
            }
        
            ,
            {
                "card_title": "宋茜回应没喝中药",
                "linkurl": "https%3A/%2Fwww.baidu.com%2Fs%3Fwd%3D%25E5%25AE%258B%25E8%258C%259C%25E5%259B%259E%25E5%25BA%2594%25E6%25B2%25A1%25E5%2596%259D%25E4%25B8%25AD%25E8%258D%25AF%26sa%3Dfyb_n_homepage%26rsv_dl%3Dfyb_n_homepage%26from%3Dsuper%26cl%3D3%26tn%3Dbaidutop10%26fr%3Dtop1000%26rsv_idx%3D2%26hisfilter%3D1",
                "views": "",
                "isViewed": "",
                "isNew": "",
                "heat_score": "5207904",
                "hotTags": "0",
                "pre_tag": "",
                "index": "28"
            }
        
            ,
            {
                "card_title": "商务部：中方已起诉美方加征关税措施",
                "linkurl": "https%3A/%2Fwww.baidu.com%2Fs%3Fwd%3D%25E5%2595%2586%25E5%258A%25A1%25E9%2583%25A8%25EF%25BC%259A%25E4%25B8%25AD%25E6%2596%25B9%25E5%25B7%25B2%25E8%25B5%25B7%25E8%25AF%2589%25E7%25BE%258E%25E6%2596%25B9%25E5%258A%25A0%25E5%25BE%2581%25E5%2585%25B3%25E7%25A8%258E%25E6%258E%25AA%25E6%2596%25BD%26sa%3Dfyb_n_homepage%26rsv_dl%3Dfyb_n_homepage%26from%3Dsuper%26cl%3D3%26tn%3Dbaidutop10%26fr%3Dtop1000%26rsv_idx%3D2%26hisfilter%3D1",
                "views": "",
                "isViewed": "",
                "isNew": "",
                "heat_score": "5111454",
                "hotTags": "0",
                "pre_tag": "",
                "index": "29"
            }
        
            ,
            {
                "card_title": "歼-16女飞行员驾驶战机一飞冲天",
                "linkurl": "https%3A/%2Fwww.baidu.com%2Fs%3Fwd%3D%25E6%25AD%25BC-16%25E5%25A5%25B3%25E9%25A3%259E%25E8%25A1%258C%25E5%2591%2598%25E9%25A9%25BE%25E9%25A9%25B6%25E6%2588%2598%25E6%259C%25BA%25E4%25B8%2580%25E9%25A3%259E%25E5%2586%25B2%25E5%25A4%25A9%26sa%3Dfyb_n_homepage%26rsv_dl%3Dfyb_n_homepage%26from%3Dsuper%26cl%3D3%26tn%3Dbaidutop10%26fr%3Dtop1000%26rsv_idx%3D2%26hisfilter%3D1",
                "views": "",
                "isViewed": "",
                "isNew": "",
                "heat_score": "5072259",
                "hotTags": "0",
                "pre_tag": "",
                "index": "30"
            }
        
            ,
            {
                "card_title": "赵露思因为躯体化记不住台词",
                "linkurl": "https%3A/%2Fwww.baidu.com%2Fs%3Fwd%3D%25E8%25B5%25B5%25E9%259C%25B2%25E6%2580%259D%25E5%259B%25A0%25E4%25B8%25BA%25E8%25BA%25AF%25E4%25BD%2593%25E5%258C%2596%25E8%25AE%25B0%25E4%25B8%258D%25E4%25BD%258F%25E5%258F%25B0%25E8%25AF%258D%26sa%3Dfyb_n_homepage%26rsv_dl%3Dfyb_n_homepage%26from%3Dsuper%26cl%3D3%26tn%3Dbaidutop10%26fr%3Dtop1000%26rsv_idx%3D2%26hisfilter%3D1",
                "views": "",
                "isViewed": "",
                "isNew": "",
                "heat_score": "4939504",
                "hotTags": "0",
                "pre_tag": "",
                "index": "31"
            }
        
            ,
            {
                "card_title": "年薪百万副总裁伪造公章被立案",
                "linkurl": "https%3A/%2Fwww.baidu.com%2Fs%3Fwd%3D%25E5%25B9%25B4%25E8%2596%25AA%25E7%2599%25BE%25E4%25B8%2587%25E5%2589%25AF%25E6%2580%25BB%25E8%25A3%2581%25E4%25BC%25AA%25E9%2580%25A0%25E5%2585%25AC%25E7%25AB%25A0%25E8%25A2%25AB%25E7%25AB%258B%25E6%25A1%2588%26sa%3Dfyb_n_homepage%26rsv_dl%3Dfyb_n_homepage%26from%3Dsuper%26cl%3D3%26tn%3Dbaidutop10%26fr%3Dtop1000%26rsv_idx%3D2%26hisfilter%3D1",
                "views": "",
                "isViewed": "",
                "isNew": "",
                "heat_score": "4850020",
                "hotTags": "0",
                "pre_tag": "",
                "index": "32"
            }
        
            ,
            {
                "card_title": "69岁阿姨自驾环游中国走出抑郁",
                "linkurl": "https%3A/%2Fwww.baidu.com%2Fs%3Fwd%3D69%25E5%25B2%2581%25E9%2598%25BF%25E5%25A7%25A8%25E8%2587%25AA%25E9%25A9%25BE%25E7%258E%25AF%25E6%25B8%25B8%25E4%25B8%25AD%25E5%259B%25BD%25E8%25B5%25B0%25E5%2587%25BA%25E6%258A%2591%25E9%2583%2581%26sa%3Dfyb_n_homepage%26rsv_dl%3Dfyb_n_homepage%26from%3Dsuper%26cl%3D3%26tn%3Dbaidutop10%26fr%3Dtop1000%26rsv_idx%3D2%26hisfilter%3D1",
                "views": "",
                "isViewed": "",
                "isNew": "",
                "heat_score": "4756465",
                "hotTags": "0",
                "pre_tag": "",
                "index": "33"
            }
        
            ,
            {
                "card_title": "刘学义《念无双》完全就是“男妲己”",
                "linkurl": "https%3A/%2Fwww.baidu.com%2Fs%3Fwd%3D%25E5%2588%2598%25E5%25AD%25A6%25E4%25B9%2589%25E3%2580%258A%25E5%25BF%25B5%25E6%2597%25A0%25E5%258F%258C%25E3%2580%258B%25E5%25AE%258C%25E5%2585%25A8%25E5%25B0%25B1%25E6%2598%25AF%25E2%2580%259C%25E7%2594%25B7%25E5%25A6%25B2%25E5%25B7%25B1%25E2%2580%259D%26sa%3Dfyb_n_homepage%26rsv_dl%3Dfyb_n_homepage%26from%3Dsuper%26cl%3D3%26tn%3Dbaidutop10%26fr%3Dtop1000%26rsv_idx%3D2%26hisfilter%3D1",
                "views": "",
                "isViewed": "",
                "isNew": "",
                "heat_score": "4602752",
                "hotTags": "0",
                "pre_tag": "",
                "index": "34"
            }
        
            ,
            {
                "card_title": "国防部：G7不要总是想着给人当教师爷",
                "linkurl": "https%3A/%2Fwww.baidu.com%2Fs%3Fwd%3D%25E5%259B%25BD%25E9%2598%25B2%25E9%2583%25A8%25EF%25BC%259AG7%25E4%25B8%258D%25E8%25A6%2581%25E6%2580%25BB%25E6%2598%25AF%25E6%2583%25B3%25E7%259D%2580%25E7%25BB%2599%25E4%25BA%25BA%25E5%25BD%2593%25E6%2595%2599%25E5%25B8%2588%25E7%2588%25B7%26sa%3Dfyb_n_homepage%26rsv_dl%3Dfyb_n_homepage%26from%3Dsuper%26cl%3D3%26tn%3Dbaidutop10%26fr%3Dtop1000%26rsv_idx%3D2%26hisfilter%3D1",
                "views": "",
                "isViewed": "",
                "isNew": "",
                "heat_score": "4529476",
                "hotTags": "0",
                "pre_tag": "",
                "index": "35"
            }
        
            ,
            {
                "card_title": "第一批挣外国人钱的年轻人已不上班",
                "linkurl": "https%3A/%2Fwww.baidu.com%2Fs%3Fwd%3D%25E7%25AC%25AC%25E4%25B8%2580%25E6%2589%25B9%25E6%258C%25A3%25E5%25A4%2596%25E5%259B%25BD%25E4%25BA%25BA%25E9%2592%25B1%25E7%259A%2584%25E5%25B9%25B4%25E8%25BD%25BB%25E4%25BA%25BA%25E5%25B7%25B2%25E4%25B8%258D%25E4%25B8%258A%25E7%258F%25AD%26sa%3Dfyb_n_homepage%26rsv_dl%3Dfyb_n_homepage%26from%3Dsuper%26cl%3D3%26tn%3Dbaidutop10%26fr%3Dtop1000%26rsv_idx%3D2%26hisfilter%3D1",
                "views": "",
                "isViewed": "",
                "isNew": "",
                "heat_score": "4493129",
                "hotTags": "0",
                "pre_tag": "",
                "index": "36"
            }
        
            ,
            {
                "card_title": "夫妻开店遭房东刁难亏本后续",
                "linkurl": "https%3A/%2Fwww.baidu.com%2Fs%3Fwd%3D%25E5%25A4%25AB%25E5%25A6%25BB%25E5%25BC%2580%25E5%25BA%2597%25E9%2581%25AD%25E6%2588%25BF%25E4%25B8%259C%25E5%2588%2581%25E9%259A%25BE%25E4%25BA%258F%25E6%259C%25AC%25E5%2590%258E%25E7%25BB%25AD%26sa%3Dfyb_n_homepage%26rsv_dl%3Dfyb_n_homepage%26from%3Dsuper%26cl%3D3%26tn%3Dbaidutop10%26fr%3Dtop1000%26rsv_idx%3D2%26hisfilter%3D1",
                "views": "",
                "isViewed": "",
                "isNew": "",
                "heat_score": "4344058",
                "hotTags": "0",
                "pre_tag": "",
                "index": "37"
            }
        
            ,
            {
                "card_title": "张雨霏恢复训练像身体被掏空",
                "linkurl": "https%3A/%2Fwww.baidu.com%2Fs%3Fwd%3D%25E5%25BC%25A0%25E9%259B%25A8%25E9%259C%258F%25E6%2581%25A2%25E5%25A4%258D%25E8%25AE%25AD%25E7%25BB%2583%25E5%2583%258F%25E8%25BA%25AB%25E4%25BD%2593%25E8%25A2%25AB%25E6%258E%258F%25E7%25A9%25BA%26sa%3Dfyb_n_homepage%26rsv_dl%3Dfyb_n_homepage%26from%3Dsuper%26cl%3D3%26tn%3Dbaidutop10%26fr%3Dtop1000%26rsv_idx%3D2%26hisfilter%3D1",
                "views": "",
                "isViewed": "",
                "isNew": "",
                "heat_score": "4235422",
                "hotTags": "0",
                "pre_tag": "",
                "index": "38"
            }
        
            ,
            {
                "card_title": "女子减肥上瘾身高1米6体重仅52斤",
                "linkurl": "https%3A/%2Fwww.baidu.com%2Fs%3Fwd%3D%25E5%25A5%25B3%25E5%25AD%2590%25E5%2587%258F%25E8%2582%25A5%25E4%25B8%258A%25E7%2598%25BE%25E8%25BA%25AB%25E9%25AB%25981%25E7%25B1%25B36%25E4%25BD%2593%25E9%2587%258D%25E4%25BB%258552%25E6%2596%25A4%26sa%3Dfyb_n_homepage%26rsv_dl%3Dfyb_n_homepage%26from%3Dsuper%26cl%3D3%26tn%3Dbaidutop10%26fr%3Dtop1000%26rsv_idx%3D2%26hisfilter%3D1",
                "views": "",
                "isViewed": "",
                "isNew": "",
                "heat_score": "4127767",
                "hotTags": "0",
                "pre_tag": "",
                "index": "39"
            }
        
            ,
            {
                "card_title": "特朗普：对所有进口汽车征收25%关税",
                "linkurl": "https%3A/%2Fwww.baidu.com%2Fs%3Fwd%3D%25E7%2589%25B9%25E6%259C%2597%25E6%2599%25AE%25EF%25BC%259A%25E5%25AF%25B9%25E6%2589%2580%25E6%259C%2589%25E8%25BF%259B%25E5%258F%25A3%25E6%25B1%25BD%25E8%25BD%25A6%25E5%25BE%2581%25E6%2594%25B625%2525%25E5%2585%25B3%25E7%25A8%258E%26sa%3Dfyb_n_homepage%26rsv_dl%3Dfyb_n_homepage%26from%3Dsuper%26cl%3D3%26tn%3Dbaidutop10%26fr%3Dtop1000%26rsv_idx%3D2%26hisfilter%3D1",
                "views": "",
                "isViewed": "",
                "isNew": "",
                "heat_score": "4085298",
                "hotTags": "3",
                "pre_tag": "",
                "index": "40"
            }
        
            ,
            {
                "card_title": "卡尼耍狠：无法与中国共享贸易价值观",
                "linkurl": "https%3A/%2Fwww.baidu.com%2Fs%3Fwd%3D%25E5%258D%25A1%25E5%25B0%25BC%25E8%2580%258D%25E7%258B%25A0%25EF%25BC%259A%25E6%2597%25A0%25E6%25B3%2595%25E4%25B8%258E%25E4%25B8%25AD%25E5%259B%25BD%25E5%2585%25B1%25E4%25BA%25AB%25E8%25B4%25B8%25E6%2598%2593%25E4%25BB%25B7%25E5%2580%25BC%25E8%25A7%2582%26sa%3Dfyb_n_homepage%26rsv_dl%3Dfyb_n_homepage%26from%3Dsuper%26cl%3D3%26tn%3Dbaidutop10%26fr%3Dtop1000%26rsv_idx%3D2%26hisfilter%3D1",
                "views": "",
                "isViewed": "",
                "isNew": "",
                "heat_score": "3937018",
                "hotTags": "0",
                "pre_tag": "",
                "index": "41"
            }
        
            ,
            {
                "card_title": "男子走进寺庙 把香灰倒出来观察香炉",
                "linkurl": "https%3A/%2Fwww.baidu.com%2Fs%3Fwd%3D%25E7%2594%25B7%25E5%25AD%2590%25E8%25B5%25B0%25E8%25BF%259B%25E5%25AF%25BA%25E5%25BA%2599%2B%25E6%258A%258A%25E9%25A6%2599%25E7%2581%25B0%25E5%2580%2592%25E5%2587%25BA%25E6%259D%25A5%25E8%25A7%2582%25E5%25AF%259F%25E9%25A6%2599%25E7%2582%2589%26sa%3Dfyb_n_homepage%26rsv_dl%3Dfyb_n_homepage%26from%3Dsuper%26cl%3D3%26tn%3Dbaidutop10%26fr%3Dtop1000%26rsv_idx%3D2%26hisfilter%3D1",
                "views": "",
                "isViewed": "",
                "isNew": "",
                "heat_score": "3827745",
                "hotTags": "0",
                "pre_tag": "",
                "index": "42"
            }
        
            ,
            {
                "card_title": "2小时车程变2分钟是什么体验",
                "linkurl": "https%3A/%2Fwww.baidu.com%2Fs%3Fwd%3D2%25E5%25B0%258F%25E6%2597%25B6%25E8%25BD%25A6%25E7%25A8%258B%25E5%258F%25982%25E5%2588%2586%25E9%2592%259F%25E6%2598%25AF%25E4%25BB%2580%25E4%25B9%2588%25E4%25BD%2593%25E9%25AA%258C%26sa%3Dfyb_n_homepage%26rsv_dl%3Dfyb_n_homepage%26from%3Dsuper%26cl%3D3%26tn%3Dbaidutop10%26fr%3Dtop1000%26rsv_idx%3D2%26hisfilter%3D1",
                "views": "",
                "isViewed": "",
                "isNew": "",
                "heat_score": "3785792",
                "hotTags": "0",
                "pre_tag": "",
                "index": "43"
            }
        
            ,
            {
                "card_title": "966万元大奖过期前被“压哨”领走",
                "linkurl": "https%3A/%2Fwww.baidu.com%2Fs%3Fwd%3D966%25E4%25B8%2587%25E5%2585%2583%25E5%25A4%25A7%25E5%25A5%2596%25E8%25BF%2587%25E6%259C%259F%25E5%2589%258D%25E8%25A2%25AB%25E2%2580%259C%25E5%258E%258B%25E5%2593%25A8%25E2%2580%259D%25E9%25A2%2586%25E8%25B5%25B0%26sa%3Dfyb_n_homepage%26rsv_dl%3Dfyb_n_homepage%26from%3Dsuper%26cl%3D3%26tn%3Dbaidutop10%26fr%3Dtop1000%26rsv_idx%3D2%26hisfilter%3D1",
                "views": "",
                "isViewed": "",
                "isNew": "",
                "heat_score": "3673285",
                "hotTags": "0",
                "pre_tag": "",
                "index": "44"
            }
        
            ,
            {
                "card_title": "高二女生用扭扭棒复刻明凤冠",
                "linkurl": "https%3A/%2Fwww.baidu.com%2Fs%3Fwd%3D%25E9%25AB%2598%25E4%25BA%258C%25E5%25A5%25B3%25E7%2594%259F%25E7%2594%25A8%25E6%2589%25AD%25E6%2589%25AD%25E6%25A3%2592%25E5%25A4%258D%25E5%2588%25BB%25E6%2598%258E%25E5%2587%25A4%25E5%2586%25A0%26sa%3Dfyb_n_homepage%26rsv_dl%3Dfyb_n_homepage%26from%3Dsuper%26cl%3D3%26tn%3Dbaidutop10%26fr%3Dtop1000%26rsv_idx%3D2%26hisfilter%3D1",
                "views": "",
                "isViewed": "",
                "isNew": "",
                "heat_score": "3596768",
                "hotTags": "0",
                "pre_tag": "",
                "index": "45"
            }
        
            ,
            {
                "card_title": "小鹏汽车回应屏幕夹断儿童手指事件",
                "linkurl": "https%3A/%2Fwww.baidu.com%2Fs%3Fwd%3D%25E5%25B0%258F%25E9%25B9%258F%25E6%25B1%25BD%25E8%25BD%25A6%25E5%259B%259E%25E5%25BA%2594%25E5%25B1%258F%25E5%25B9%2595%25E5%25A4%25B9%25E6%2596%25AD%25E5%2584%25BF%25E7%25AB%25A5%25E6%2589%258B%25E6%258C%2587%25E4%25BA%258B%25E4%25BB%25B6%26sa%3Dfyb_n_homepage%26rsv_dl%3Dfyb_n_homepage%26from%3Dsuper%26cl%3D3%26tn%3Dbaidutop10%26fr%3Dtop1000%26rsv_idx%3D2%26hisfilter%3D1",
                "views": "",
                "isViewed": "",
                "isNew": "",
                "heat_score": "3471502",
                "hotTags": "0",
                "pre_tag": "",
                "index": "46"
            }
        
            ,
            {
                "card_title": "恒大汽车回应股价一度暴涨超230%",
                "linkurl": "https%3A/%2Fwww.baidu.com%2Fs%3Fwd%3D%25E6%2581%2592%25E5%25A4%25A7%25E6%25B1%25BD%25E8%25BD%25A6%25E5%259B%259E%25E5%25BA%2594%25E8%2582%25A1%25E4%25BB%25B7%25E4%25B8%2580%25E5%25BA%25A6%25E6%259A%25B4%25E6%25B6%25A8%25E8%25B6%2585230%2525%26sa%3Dfyb_n_homepage%26rsv_dl%3Dfyb_n_homepage%26from%3Dsuper%26cl%3D3%26tn%3Dbaidutop10%26fr%3Dtop1000%26rsv_idx%3D2%26hisfilter%3D1",
                "views": "",
                "isViewed": "",
                "isNew": "",
                "heat_score": "3376632",
                "hotTags": "0",
                "pre_tag": "",
                "index": "47"
            }
        
            ,
            {
                "card_title": "南京高科一项目“割肉”促销",
                "linkurl": "https%3A/%2Fwww.baidu.com%2Fs%3Fwd%3D%25E5%258D%2597%25E4%25BA%25AC%25E9%25AB%2598%25E7%25A7%2591%25E4%25B8%2580%25E9%25A1%25B9%25E7%259B%25AE%25E2%2580%259C%25E5%2589%25B2%25E8%2582%2589%25E2%2580%259D%25E4%25BF%2583%25E9%2594%2580%26sa%3Dfyb_n_homepage%26rsv_dl%3Dfyb_n_homepage%26from%3Dsuper%26cl%3D3%26tn%3Dbaidutop10%26fr%3Dtop1000%26rsv_idx%3D2%26hisfilter%3D1",
                "views": "",
                "isViewed": "",
                "isNew": "",
                "heat_score": "3211842",
                "hotTags": "0",
                "pre_tag": "",
                "index": "48"
            }
        
            ,
            {
                "card_title": "小伙靠书法帮父母还清2000万巨债",
                "linkurl": "https%3A/%2Fwww.baidu.com%2Fs%3Fwd%3D%25E5%25B0%258F%25E4%25BC%2599%25E9%259D%25A0%25E4%25B9%25A6%25E6%25B3%2595%25E5%25B8%25AE%25E7%2588%25B6%25E6%25AF%258D%25E8%25BF%2598%25E6%25B8%25852000%25E4%25B8%2587%25E5%25B7%25A8%25E5%2580%25BA%26sa%3Dfyb_n_homepage%26rsv_dl%3Dfyb_n_homepage%26from%3Dsuper%26cl%3D3%26tn%3Dbaidutop10%26fr%3Dtop1000%26rsv_idx%3D2%26hisfilter%3D1",
                "views": "",
                "isViewed": "",
                "isNew": "",
                "heat_score": "3124946",
                "hotTags": "3",
                "pre_tag": "",
                "index": "49"
            }
        ]}
    </textarea><div id="s_ctner_contents" class="s-ctner-contents   hot-news-all"><div id="s_content_99" class="s-content  hot-content" data-id="99"><!-- node渲染 --><script>window._card_aync = 1;</script><div id="s_content_99" class="s-content"><div class="s-content-load"><img src="https://pss.bdstatic.com/static/superman/img/loading_new-f4a658cef0.gif" /></div></div></div></div><textarea id="s_menus_textarea" style="display:none;">
{
    "data": [
        
        
        {
            "id": 100,
            "name": "我的关注",
            "state": "1",
            "ismenu": "0"
        }
        
        ,
        {
            "id": 2,
            "name": "推荐",
            "state": "1",
            "ismenu": "1"
        }
        
        ,
        {
            "id": 1,
            "name": "导航",
            "state": "1",
            "ismenu": "1"
        }
        
        ,
        {
            "id": 99,
            "name": "热搜",
            "state": "1",
            "ismenu": "0"
        }
        
    ],
    "isEnd": "",
    "illegal": ""
}
</textarea><div id="s_side_wrapper"><div class="side-entry aging-entry"><div class="aging-entry-inner"></div><div class="toast c-color-text">辅助模式</div></div><div id="s_qrcode_feed" class="qrcode-feed side-entry"><div class="qrcode-layer icon-mask-wrapper"><img class="icon" src="https://pss.bdstatic.com/static/superman/img/side/qrcode@2x-daf987ad02.png"/><img class="icon-hover" src="https://pss.bdstatic.com/static/superman/img/side/qrcode-hover@2x-f9b106a848.png"/></div><div class="tooltip qrcode-tooltip"><div class="text"><div class="download-text c-font-big">扫码下载百度APP</div><div class="c-font-normal c-color-gray2">百度一下&nbsp;&nbsp;生活更好</div></div><img class="qrcode" src="https://pss.bdstatic.com/static/superman/img/qrcode_download-02b84e1f66.png"/></div></div><div id="s_top_feed" class="top-feed side-entry" style="display:none;"><div class="to-top icon-mask-wrapper" data-tool="1"><img class="icon" src="https://pss.bdstatic.com/static/superman/img/side/totop@2x-baa701cb1b.png"/><img class="icon-hover" src="https://pss.bdstatic.com/static/superman/img/side/totop-hover@2x-9d6bbde373.png"/></div><div class="tooltip totop-tooltip c-font-medium c-color-text">返回顶部</div></div></div></div><script type="text/javascript">
            window._card_sync = 1;
            // 首屏图片加载监控
            window._xman_speed && _xman_speed.imgLoadTime(99);
        </script></div><textarea id="s_sync_data" style="display:none;"></textarea></div><script type="application/json" id="promote_login_box">
    []
</script><div id="bottom_layer" class="s-bottom-layer s-isindex-wrap"><div class="s-bottom-layer-content"><p class="lh"><a class="text-color" href="//home.baidu.com" target="_blank">关于百度</a></p><p class="lh"><a class="text-color" href="http://ir.baidu.com" target="_blank">About Baidu</a></p><p class="lh"><a class="text-color" href="//www.baidu.com/duty" target="_blank">使用百度前必读</a></p><p class="lh"><a class="text-color" href="https://help.baidu.com/question?prod_id=1" target="_blank">帮助中心</a></p><p class="lh"><a class="text-color" href="https://e.baidu.com/?refer=1271" target="_blank">企业推广</a></p><p class="lh"><a class="text-color" href="http://www.beian.gov.cn/portal/registerSystemInfo?recordcode=11000002000001" target="_blank">京公网安备11000002000001号</a></p><p class="lh"><a class="text-color" href="https://beian.miit.gov.cn" target="_blank">京ICP证030173号</a></p><p class="lh"><span class="text-color">互联网新闻信息服务许可证11220180008</span></p><p class="lh"><span class="text-color">网络文化经营许可证： 京网文〔2023〕1034-029号</span></p><p class="lh"><a class="text-color" href="//www.baidu.com/licence/" target="_blank">信息网络传播视听节目许可证 0110516</a></p><p class="lh"><span class="text-color">互联网宗教信息服务许可证编号：京（2022）0000043</span></p><p class="lh"><span class="text-color">药品医疗器械网络信息服务备案（京）网药械信息备字（2021）第00159号</span></p><p class="lh"><span class="text-color">医疗器械网络交易服务第三方平台备案凭证（京）网械平台备字（2020）第00002号</span></p><p class="lh"><span class="text-color">药品网络交易服务第三方平台备案凭证（京）网药平台备字〔2023〕第000002号</span></p><p class="lh"><span class="text-color">&#169;2025&nbsp;Baidu&nbsp;</span></p></div></div></div><div class="s_tab " id="s_tab"><div class="s_tab_inner pc_home_inner_tab  s-old-tag "><b class="cur-tab">网页</b><a href="https://chat.baidu.com/search?pd=csaitab&amp;setype=csaitab&amp;extParamsJson=%7B%22enter_type%22%3A%22search_a_tab%22%2C%22sa%22%3A%22vs_tab%22%7D"  wdfield="word"  onmousedown="return c({'fm':'tab','tab':'csaitab'})"  class="s-tab-item  s-tab-item-img"><img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAE4AAAAqCAMAAAAqEZ1jAAAAAXNSR0IArs4c6QAAAAlwSFlzAAAhOAAAITgBRZYxYAAAAKJQTFRFAAAAenb/RpP/k2j/XXf7cXX6P5z7i2n8W3j7ZHP7jmn9VID7Qpf8jWn8Xnr6iWr8YHb7i2n8RJb9dWz8WXr8mWj9RpP8W3n8bW/8l2j+QJz9Q5f9ToT8fWv9ZW/8lWj9VID8YnX7Pp79QZr9pWb+RJT8m2f9k2j9SYz8jWn9hmr8UIT8fmv8VX78Xnn7d2z8ZHT7WHr7bm77Xnb7Z3D7YHH7RJOQRAAAACJ0Uk5TABAgICAwQEBAWF5gZXBwgICbn5+fo7+/vsLP39/f3urv73XwOA8AAAKfSURBVHja7dbJcuIwFIXhIzCxMTMNcdwMDoMZY4NxeP9X66srEckYQlPVvcvPBhZ8dVSIAvz03xNV/Luq4Xq9Hgo8nfDqjXq97hS1yXZNhXg2Z3o+f1KnLqx62y17Pp5smp+158EUaW6I52rkxLF3eoNpt1Ne+bR+Fd8U5Oc8Px6P5M0Evgq11ytpy0jgbk6e5cSxd2pb7yKOikpTouWyhbsNMsnpefZpOzsCI7c8brkM74/LqFSC7HkwVTu9lsB14ZJy745LlDdV5+3iQTXC7s9z3pNEggOh9s3woOFqJT2BmzUSLnMQEEeeh2+rkraiOrjZeL+XXEBwyp71Ybg+hWL+iltGt8ftKQI9QMzSlDxz9VofssIMdxitdJEvbo3jxqAGKXtt6N4OkpvgkmiFa50Ch1UU8zab/Ya4Br/IUnnewHCHg+FEh7/ExUIXdq+KG4N7z3hfzXDkaa5FmGpdMENrYWWjaoDrJtJLu4ajFFfbUduryN1ue4brb2KpjSvgvETe53R24U6Ggz+RYJnsCTMujmMJ9qELErkv9y7cyXAGtNCoI2Dqx9ymAl1bcYMLJ2NO5fZ2dmHLxiBGrJlxEAl7M1HkTFUzseeiWHOxYK+CrwLltRX3qTk7/4PbTXDdaBHLxytMdcllaaA56U1RSEyU56M0jotfYBLvynM091ni0GGtPO73nLkR7AaK6yqOKnG1g+wXrnqZz9nrw87bszdjjn/dDGffbgdX9YmT4HykaoLTp/UABGfyypx7a1yFNb0wpl7BddW8geSO5JU5TE63xhmOPb2uok8reN1Nrn4qjQNBlmfdvkB5beaoMgdP4LpFgbNuX3svtSwABoZ7WPOKa0Inxgl7DurMdfFXVV7savjK8WQ1/cz5+Q9e7A/jUZeiPQO0fwAAAABJRU5ErkJggg=="/></a><a href="http://image.baidu.com/i?tn=baiduimage&amp;ps=1&amp;ct=201326592&amp;lm=-1&amp;cl=2&amp;nc=1&amp;ie=utf-8"  wdfield="word"  onmousedown="return c({'fm':'tab','tab':'pic'})"  class="s-tab-item  s-tab-pic">
                    
                        图片
                    
                </a><a href="https://www.baidu.com/s?rtt=1&amp;bsst=1&amp;cl=2&amp;tn=news&amp;ie=utf-8"  wdfield="word"  onmousedown="return c({'fm':'tab','tab':'news'})" sync="true" class="s-tab-item  s-tab-news">
                    
                        资讯
                    
                </a><a href="http://v.baidu.com/v?ct=301989888&amp;rn=20&amp;pn=0&amp;db=0&amp;s=25&amp;ie=utf-8"  wdfield="word"  onmousedown="return c({'fm':'tab','tab':'video'})"  class="s-tab-item  s-tab-video">
                    
                        视频
                    
                </a><a href="http://www.baidu.com/s?pd=note&amp;rpf=pc"  wdfield="word"  onmousedown="return c({'fm':'tab','tab':'note'})"  class="s-tab-item  s-tab-note">
                    
                        笔记
                    
                </a><a href="https://map.baidu.com/?newmap=1&amp;ie=utf-8&amp;from=pstab&amp;s=s"   onmousedown="return c({'fm':'tab','tab':'map'})"  class="s-tab-item  s-tab-map">
                    
                        地图
                    
                </a><a href="http://tieba.baidu.com/f?fr=wwwt&amp;ie=utf-8"  wdfield="kw"  onmousedown="return c({'fm':'tab','tab':'tieba'})"  class="s-tab-item  s-tab-tieba">
                    
                        贴吧
                    
                </a><a href="http://wenku.baidu.com/search?lm=0&amp;od=0&amp;ie=utf-8"  wdfield="word"  onmousedown="return c({'fm':'tab','tab':'wenku'})"  class="s-tab-item  s-tab-wenku">
                    
                        文库
                    
                </a><a href="http://www.baidu.com/more/"   onmousedown="return c({'fm':'tab','tab':'more'})"  class="s-tab-item  s-tab-more">
                    
                        更多
                    
                </a></div></div><div id="wrapper_wrapper"></div></div><input type="hidden" id="bsToken" name="bsToken" value="5a07111fb6ed83692d5ceeb7a88706eb" /><script>
    
    





var bds = {
    se: {},
    su: {
        urdata:[],
        urSendClick: function(){},
        urStatic:"https://dss1.bdstatic.com/5aV1bjqh_Q23odCf"
    },
    util: {},
    use: {},
    comm: {
        domain:"",
        ubsurl:"",
        tn:"",
        queryEnc:"",
        queryId:"",
        inter:"",
        sugHost:"",
        query:"",
        qid:"",
        cid:"",
        sid:"",
        stoken:"",
        serverTime:"",
        user:"弹定的药",
        username:"弹定的药",
        loginAction:[],
        useFavo:"",
        pinyin:"",
        favoOn:"",
        curResultNum:"0",
        rightResultExist:false,
        protectNum:0,
        zxlNum:0,
        pageNum:1,
        pageSize:10,
        ishome:1,
        newindex:1
    }
};

bds.comm.indexSid = "62232";

bds.comm.personalData = 
{
    
        
        
        
        "city_weather":"0"
        
        ,
        "click_site":"a:1:{i:0;s:17:\&quot;1931363&amp;272804290\&quot;;}"
        
        ,
        "closeCardSceneRec":"0"
        
        ,
        "curCard":"99"
        
        ,
        "fullSkinName":"0"
        
        ,
        "homepageTTS":"0"
        
        ,
        "imeSwitch":"0"
        
        ,
        "isJumpHttps":"1"
        
        ,
        "isLite":"0"
        
        ,
        "isSwitch":"1"
        
        ,
        "personalSwitch":"1"
        
        ,
        "resultLang":"0"
        
        ,
        "resultNum":"10"
        
        ,
        "skinHistory":"super_150:super_44:super_31:super_30"
        
        ,
        "skinUrl":""
        
        ,
        "skinUrlHis":"super_150:super_44:super_31:super_30"
        
        ,
        "soccer":"1"
        
        ,
        "sugSet":"1"
        
        ,
        "sugStoreSet":"0"
        
        ,
        "userCards":"{\&quot;8\&quot;:\&quot;on\&quot;}"
        
        ,
        "worldcup_extra":"1"
        
    
}
;

bds.comm.dpquery="";

bds.comm.isNode = 1;

    


var name,navigate,al_arr=[];

var selfOpen = window.open;
eval("var open = selfOpen;");

var isIE = navigator.userAgent.indexOf("MSIE") != -1 && !window.opera;

var E = bds.ecom = {};

bds.se.mon = {
    loadedItems: [],
    load: function() {},
    srvt:-1
};
try { 
    bds.se.mon.srvt = parseInt(document.cookie.match(new RegExp("(^| )BDSVRTM=([^;]*)(;|$)"))[2]);
    document.cookie = "BDSVRTM=;expires=Sat, 01 Jan 2000 00:00:00 GMT"
} catch (e) {}


var
bdUser        = bds.comm.user?bds.comm.user:null,
bdQuery       = bds.comm.query,
bdUseFavo     = bds.comm.useFavo,
bdFavoOn      = bds.comm.favoOn,
bdCid         = bds.comm.cid,
bdSid         = bds.comm.sid,
bdServerTime  = bds.comm.serverTime,
bdQid         = bds.comm.queryId,
bdstoken      = bds.comm.stoken,
login_success = [];




bds.comm.sampleval= [];
bds.comm.nodeSample = ["PC_INDEX_BOX_REC_OFF"]

bds.comm.desktopSampleval = "";

bds.comm.aiSampleval = false;
bds.comm.hitHotBoardSample = false;
bds.comm.hitRecommendToolSample = false
bds.comm.hitRecommendStyleSample = false
bds.comm.hitBoxRecSample = false

bds.comm.showFrontPageSettings = false;



    bds.comm.samNewBox = 0;




bds.comm.newTopMenu = 1;
bds.comm.newSearchBox = 1;

bds.comm.newSearchbox = 1;


bds.comm.sIndex = 1;








    bds.comm.pdc = 0;



    bds.comm.newNav = 0;




    bds.comm.inputUpgrade = 0;



    bds.comm.headTabUpgrade = 0;



    bds.comm.samNewBox = 0;



bds.comm.popUpAdvert = {}





bds.comm.withFeedData = false;


bds.comm.boxRecSwitch = "";


bds.comm.pcFreshWrapper = "";


bds.comm.pcFreshCard = "";


bds.comm.pcFreshTitle = "";


bds.comm.pcFreshSmooth = "";


bds.comm.pcSugSwitch = 0;

    



var s_domain = {
    "protocol": "https",
    "staticUrl": "https://pss.bdstatic.com/",
    "base": "home",
    "baseuri": "/home",
    "passconf": "http://passport.baidu.com/ubrwsbas",
    "logout": "https://passport.baidu.com/?logout&amp;u=",
    "bs": "www.baidu.com",
    "login": "http://passport.baidu.com/?login&amp;tpl=super&amp;u=",
    "sp": "http://hi.baidu.com/",
    "ssllist": 
{
    
        
        
        
        "su.bdimg.com":"dss0.bdstatic.com\/5a21bjqh_Q23odCf"
        
        ,
        "1.ur.bdimg.com":"dss1.bdstatic.com\/k4oTfnSm1A5BphGlnYG"
        
        ,
        "2.ur.bdimg.com":"dss2.bdstatic.com\/kvoTfnSm1A5BphGlnYG"
        
        ,
        "3.ur.bdimg.com":"dss3.bdstatic.com\/kfoTfnSm1A5BphGlnYG"
        
        ,
        "4.ur.bdimg.com":"dss0.bdstatic.com\/lPoTfnSm1A5BphGlnYG"
        
        ,
        "5.ur.bdimg.com":"dss1.bdstatic.com\/l4oTfnSm1A5BphGlnYG"
        
        ,
        "6.ur.bdimg.com":"dss2.bdstatic.com\/lvoTfnSm1A5BphGlnYG"
        
        ,
        "7.ur.bdimg.com":"dss3.bdstatic.com\/lfoTfnSm1A5BphGlnYG"
        
        ,
        "8.ur.bdimg.com":"dss0.bdstatic.com\/iPoTfnSm1A5BphGlnYG"
        
        ,
        "dj0.baidu.com":"sp1.baidu.com\/-L-Wsjip0QIZ8tyhnq"
        
        ,
        "dj1.baidu.com":"sp1.baidu.com\/-L-Xsjip0QIZ8tyhnq"
        
        ,
        "dj2.baidu.com":"sp2.baidu.com\/-L-Ysjip0QIZ8tyhnq"
        
        ,
        "eclick.baidu.com":"sp3.baidu.com\/-0U_dTmfKgQFm2e88IuM_a"
        
        ,
        "a.hiphotos.baidu.com":"dss0.baidu.com\/94o3dSag_xI4khGko9WTAnF6hhy"
        
        ,
        "b.hiphotos.baidu.com":"dss1.baidu.com\/9vo3dSag_xI4khGko9WTAnF6hhy"
        
        ,
        "c.hiphotos.baidu.com":"dss3.baidu.com\/9fo3dSag_xI4khGko9WTAnF6hhy"
        
        ,
        "d.hiphotos.baidu.com":"dss0.baidu.com\/-Po3dSag_xI4khGko9WTAnF6hhy"
        
        ,
        "e.hiphotos.baidu.com":"dss1.baidu.com\/-4o3dSag_xI4khGko9WTAnF6hhy"
        
        ,
        "f.hiphotos.baidu.com":"dss2.baidu.com\/-vo3dSag_xI4khGko9WTAnF6hhy"
        
        ,
        "g.hiphotos.baidu.com":"dss3.baidu.com\/-fo3dSag_xI4khGko9WTAnF6hhy"
        
        ,
        "h.hiphotos.baidu.com":"dss0.baidu.com\/7Po3dSag_xI4khGko9WTAnF6hhy"
        
        ,
        "bdimg.share.baidu.com":"dss1.baidu.com\/9rA4cT8aBw9FktbgoI7O1ygwehsv"
        
        ,
        "s.share.baidu.com":"sp1.baidu.com\/5foZdDe71MgCo2Kml5_Y_D3"
        
        ,
        "s1.bdstatic.com":"dss1.bdstatic.com\/5eN1bjq8AAUYm2zgoY3K"
        
        ,
        "p2.youxi.bdimg.com":"dss2.bdstatic.com\/5OZ1hTW64A63otebn9fN2DJv"
        
        ,
        "youxi.baidu.com":"sp3.baidu.com\/y0kThD4a2gU2pMbgoY3K"
        
        ,
        "fm.baidu.com":"dss3.baidu.com\/-rd1bjeh1BF3odCf"
        
        ,
        "music.baidu.com":"dss2.baidu.com\/8_1ZdTna2gU2pMbgoY3K"
        
        ,
        "f3.baidu.com":"sp2.baidu.com\/-uV1bjeh1BF3odCf"
        
        ,
        "sclick.baidu.com":"sp1.baidu.com\/5bU_dTmfKgQFm2e88IuM_a"
        
        ,
        "1.su.bdimg.com":"dss0.bdstatic.com\/k4oZeXSm1A5BphGlnYG"
        
        ,
        "2.su.bdimg.com":"dss1.bdstatic.com\/kvoZeXSm1A5BphGlnYG"
        
        ,
        "3.su.bdimg.com":"dss2.bdstatic.com\/kfoZeXSm1A5BphGlnYG"
        
        ,
        "4.su.bdimg.com":"dss3.bdstatic.com\/lPoZeXSm1A5BphGlnYG"
        
        ,
        "5.su.bdimg.com":"dss0.bdstatic.com\/l4oZeXSm1A5BphGlnYG"
        
        ,
        "6.su.bdimg.com":"dss1.bdstatic.com\/lvoZeXSm1A5BphGlnYG"
        
        ,
        "7.su.bdimg.com":"dss2.bdstatic.com\/lfoZeXSm1A5BphGlnYG"
        
        ,
        "8.su.bdimg.com":"dss3.bdstatic.com\/iPoZeXSm1A5BphGlnYG"
        
        ,
        "nssug.baidu.com":"sp1.baidu.com\/8qUZeT8a2gU2pMbgoY3K"
        
        ,
        "up.photo.baidu.com":"sp1.baidu.com\/6_R1fD_bAAd3otqbppnN2DJv"
        
        ,
        "suggestion.baidu.com":"sp1.baidu.com\/5a1Fazu8AA54nxGko9WTAnF6hhy"
        
        ,
        "t10.baidu.com":"t10.baidu.com"
        
        ,
        "t11.baidu.com":"t11.baidu.com"
        
        ,
        "t12.baidu.com":"t12.baidu.com"
        
        ,
        "play.baidu.com":"dss3.baidu.com\/5LgHhXSm2Q5IlBGlnYG"
        
        ,
        "olime.baidu.com":"sp1.baidu.com\/8bg4cTva2gU2pMbgoY3K"
        
        ,
        "i.baidu.com":"sp1.baidu.com\/74oIbT3kAMgDnd_"
        
        ,
        "c.baidu.com":"sp1.baidu.com\/9foIbT3kAMgDnd_"
        
        ,
        "b1.baidu.com":"dss1.baidu.com\/9uN1bjeh1BF3odCf"
        
        ,
        "nsclick.baidu.com":"sp1.baidu.com\/8qUJcD3n0sgCo2Kml5_Y_D3"
        
        ,
        "b1.bdstatic.com":"dss0.bdstatic.com\/9uN1bjq8AAUYm2zgoY3K"
        
        ,
        "i7.baidu.com":"dss0.baidu.com\/73F1bjeh1BF3odCf"
        
        ,
        "i8.baidu.com":"dss0.baidu.com\/73x1bjeh1BF3odCf"
        
        ,
        "i9.baidu.com":"dss0.baidu.com\/73t1bjeh1BF3odCf"
        
        ,
        "ecma.bdimg.com":"dss1.bdstatic.com\/-0U0bXSm1A5BphGlnYG"
        
        ,
        "tag.baidu.com":"sp1.baidu.com\/6LMFsjip0QIZ8tyhnq"
        
        ,
        "hw.baidu.com":"sp1.baidu.com\/7KF1bjeh1BF3odCf"
        
        ,
        "opendata.baidu.com":"sp1.baidu.com\/8aQDcjqpAAV3otqbppnN2DJv"
        
        ,
        "open.baidu.com":"dss1.baidu.com\/8aQDcnSm2Q5IlBGlnYG"
        
        ,
        "api.open.baidu.com":"sp1.baidu.com\/9_Q4sjW91Qh3otqbppnN2DJv"
        
        ,
        "xiaodu.baidu.com":"sp1.baidu.com\/yLsHczq6KgQFm2e88IuM_a"
        
        ,
        "s0.nuomi.bdimg.com":"dss1.bdstatic.com\/5eR1ciub_Q63otebn9fN2DJv"
        
        ,
        "s1.nuomi.bdimg.com":"dss0.baidu.com\/5eN1ciub_Q63otebn9fN2DJv"
        
        ,
        "s2.nuomi.bdimg.com":"dss2.bdstatic.com\/5eZ1ciub_Q63otebn9fN2DJv"
        
        ,
        "vse.baidu.com":"sp3.baidu.com\/6qUDsjip0QIZ8tyhnq"
        
        ,
        "himg.bdimg.com":"dss1.bdstatic.com\/7Ls0a8Sm1A5BphGlnYG"
        
        ,
        "ss.bdimg.com":"dss1.bdstatic.com\/5aV1bjqh_Q23odCf"
        
        ,
        "ecmb.bdimg.com":"dss0.bdstatic.com\/-0U0bnSm1A5BphGlnYG"
        
        ,
        "e.su.bdimg.com":"dss0.bdstatic.com\/-4oZeXSm1A5BphGlnYG"
        
        ,
        "sensearch.baidu.com":"sp1.baidu.com\/5b11fzupBgM18t7jm9iCKT-xh_"
        
        ,
        "sestat.baidu.com":"sp1.baidu.com\/5b1ZeDe5KgQFm2e88IuM_a"
        
        ,
        "cdn00.baidu-img.cn":"dss0.bdstatic.com\/9bA1vGba2gU2pMbfm9GUKT-w"
        
        ,
        "cdn01.baidu-img.cn":"dss0.bdstatic.com\/9bA1vGfa2gU2pMbfm9GUKT-w"
        
    
}

};


    var s_session={
        "logId":"2150081325",
        "sessionId":"",
        "rids":"",
        "seqId":"0xfd59a6e40027a32d",
        "qid":"0xfd59a6e40027a32d",
        "debug":false,
        "portrait":"public.1.32197c18.0XYvCY6zrzOAiO6EuOEwag?_d=29051186",
        "logPortrait":"",
        "portraitSkin2nd":"499be5bcb9e5ae9ae79a84e88dafd509",
        "userProp":
{
    
}
,
        "userTips":
{
    
        
        
            
            "hasWeatherMod":"1"
        
            ,
            "homepageTTS":"0"
        
            ,
            "isLite":"0"
        
            ,
            "isNewTabSearch":"0"
        
            ,
            "randomSkin":"0"
        
            ,
            "skinLogoHasSkined":"1"
        
            ,
            "weatherShowWarning":"1"
        
    
}
,
        "userAttr":Number("")|| 0,
        "username":"弹定的药",
        "unametype":"1",
        "userIsSkined":"off",
        "userIsNewSkined":"off",
        "userSkinName":"0",
        "userSkinOpacity":"70",
        "userSkinUrl":"",
        "userSkinLogo":"logo_white",
        "userSkinDefined":"off",
        "userLogoSrc":"\/\/www.baidu.com\/img\/PCtm_d9c8750bed0b3c7d089fa7d55720d6cf.png",
        "userLogoSrcRe":"\/\/www.baidu.com\/img\/PCtm_d9c8750bed0b3c7d089fa7d55720d6cf.png",
        "userLogoIsFestival":"1",
        "userLogoIsupload":"false",
        "usertype":"",
        "curmod": "99",
        "firstmod":"99",
        "userFirst":"",
        "logoCode":false,
        "isFesLogo":true,
        "isHaveWrodLink":false,
        "userNavEmpty":"",
        "bgUNE":"",
        "isEmptyRpRec":"",
        "userBirthday":"",
        "yuce":"",
        "sid":"60272_61027_62232_62325_62344_62346_62373_62426_62475_62484_62456_62455_62452_62450_62674_62618",
        "mSid":"",
        "sample_value":"PC_INDEX_BOX_REC_OFF",
        "strategy_hit":"",
        "issync":"",
        "isdev":"",
        "isUsedNav":"",
        "isRewardBox":"",
        "isNewMsg":"",
        "load_module":{"type":["weather"],"weatherID":''},
        "modules":["0"],
        "modules_load":["0"],
        "isLiteVersion":"0",
        "crids":"",
        "audioModernizr":false,
        "isLogin":true,
        "agingVoice": "0",
        "isLite":"0"
    };
    
    
    

    window._sp_async = 1;
    window.__async_strategy=2;
</script><script type="text/javascript" src="https://pss.bdstatic.com/static/superman/js/lib/jquery-1-edb203c114.10.2.js"></script><script type="text/javascript" src="https://pss.bdstatic.com/static/superman/js/lib/esl-d776bfb1aa.js"></script><script data-for="esl-config">(function(){var amd={keys:{},addPaths:function(opt){if(typeof opt!=="object"){return}for(var key in opt){if(opt.hasOwnProperty(key)){opt[key]=opt[key].replace(/(\.js)/,"")}}require.config({paths:opt});$.extend(amd.keys,opt)},addConfig:function(config){if(typeof config!=="object"){return}require.config(config);for(var item in config){if(config.hasOwnProperty(item)){if(!amd[item]){amd[item]={}}for(var key in config[item]){if(config[item].hasOwnProperty(key)&&config[item][key]){
amd[item][key]=config[item][key]}}}}}};s_domain.amd=amd;var staticUrl=s_domain.staticUrl+"static/superman/";amd.addPaths({"@baidu/video-meeting":staticUrl+"amd_modules/@baidu/video-meeting-1be7f62dac.js","superman/components/video-meet":staticUrl+"js/components/video-meet-7833028d86.js",tslib:staticUrl+"amd_modules/tslib-c95383af0c.js","superman/components/guide_tips":staticUrl+"js/components/guide_tips-d9e617f782.js",
"superman/components/login_guide":staticUrl+"js/components/login_guide-4fba3971ce.js","superman/components/bbox/bbox_view":staticUrl+"js/components/bbox/bbox-view-df7e8cb3b0.js","@baidu/aging-tools-pc":staticUrl+"amd_modules/@baidu/aging-tools-pc-1e5afe8bdf.js","superman/components/aging-tools":staticUrl+"js/components/aging-tools-094cd28890.js","superman/components/ubc":staticUrl+"js/components/ubc.js",
"superman/components/top-right-operate/operate":staticUrl+"js/components/top-right-operate/operate-b7693b0cc7.js","superman/components/content-info":staticUrl+"js/components/content-info-12dbf9fb6d.js","superman/components/csaitab-log":staticUrl+"js/components/csaitab-log-a9c9cae804.js","superman/components/invoke":staticUrl+"js/components/invoke-97e9694cb9.js","superman/components/placeholder":staticUrl+"js/components/placeholder-b83e2493f9.js",
"superman/components/new-search-guide-bub":staticUrl+"js/components/new-search-guide-bub-2ac19f2971.js","ai-search-box-entry":staticUrl+"/amd_modules/@baidu/ai-search-box-entry-ea20fec552.js"})})();</script><script type="text/javascript">bds.util.domain=function(){var list=$.extend({},s_domain.ssllist);$.each(list,function(i,e){if(list[i].indexOf("https://")<0){list[i]="https://"+list[i]}});var get=function(url){if(s_domain.protocol=="http:"){return url}var reg=/^(http[s]?:\/\/)?([^\/]+)(.*)/,matches=url.match(reg);url=list.hasOwnProperty(matches[2])&&list[matches[2]]+matches[3]||url;return url},set=function(kdomain,vdomain){list[kdomain]=vdomain};return{get:get,set:set}}();bds.comm.sugHost=bds.util.domain.get(
"http://suggestion.baidu.com/su");</script><script type="text/javascript">function isHitSampling(e){var t=!1;return bds&&bds.comm&&bds.comm.sampleval&&(t=bds.comm.sampleval.indexOf(e)>=0),!t&&bds&&bds.comm&&bds.comm.nodeSample&&(t=bds.comm.nodeSample.indexOf(e)>=0),t}define("modules/monitor/log-send",["require","exports"],function(e,t){"use strict";function o(e){if(!r.sample.hasOwnProperty(e))return!1;var t=r.sample[e],o=document.cookie.indexOf("webbtest=1")>-1,i=o||Math.random()<t;return r.sample[e]=i?1:0,i}function i(e){return"1"===bds.comm.alwaysMonitor?(e.alwaysMonitor=1,!0):!1}function n(e,t){if(void 0===t&&(t="except"),!(e.group&&o(e.group)||i(e.info)))return"";var n=e.pid||"1_87";bds.comm.ishome&&(n="1_79");var s=bds.comm.qid||bds.comm.queryId,a=r.logServer+"?pid="+n+"&lid="+s+"&ts="+Date.now()+"&type="+(t||e.type)+"&group="+e.group+"&info="+encodeURIComponent(JSON.stringify(e.info))+"&dim="+encodeURIComponent(JSON.stringify(e.dim||{})),c=new Image;return c.src=a,a}t.__esModule=!0,t.send=void 0;var r={pid:"1_87",sample:{jserror:.02,iframe:.02,resLoadSlow:.02,imgError:.02,resBigimg:.02,httpRes:.02,lsInfo:1,lsCost:.02,codeCoverage:.02,pcUrlUnEncode:.1,abblock:.01,ajaxError:.01,screenDist:.01,blockMonitor:.001,domMonitor:.001},logServer:"https://sp1.baidu.com/5b1ZeDe5KgQFm2e88IuM_a/mwb2.gif"};t.send=n}),define("modules/monitor/js-except",["require","exports","modules/monitor/log-send"],function(e,t,o){"use strict";function i(e,t){if(t.indexOf("chrome-extension://")>-1||t.indexOf("moz-extension://")>-1)return!1;var o=e.toLowerCase();return"script error."===o||"script error"===o?!1:!0}function n(e,t){try{var n={info:{},dim:{},group:""},r=n.info,s=e.target||e.srcElement,a=navigator.connection||{};if(r.downlink=a.downlink,r.effectiveType=a.effectiveType,r.rtt=a.rtt,r.deviceMemory=navigator.deviceMemory||0,r.hardwareConcurrency=navigator.hardwareConcurrency||0,r.saveData=!!a.saveData,s!==window&&"onerror"!==t)return;var c=e.error||{},p=c.stack||"";e.message&&i(e.message,p)&&(n.group="jserror",r.msg=e.message,r.file=e.filename,r.ln=e.lineno,r.col=e.colno,r.stack=p.split("\n").slice(0,3).join("\n"),o.send(n))}catch(d){console.error(d)}}function r(){var e,t=!1,o=navigator.userAgent.toLowerCase(),i=/msie ([0-9]+)/.exec(o);if(i&&i[1]){if(e=parseInt(i[1],10),7>=e)return;9>=e&&(t=!0)}t?window.onerror=function(e,t,o,i){n({message:e,filename:t,lineno:o,colno:i},"onerror")}:window.addEventListener&&window.addEventListener("error",n,!0)}t.__esModule=!0,t.listenerExcept=void 0,t.listenerExcept=r}),define("modules/ajax-log/ajax-log.service",["require","exports","tslib","modules/monitor/log-send"],function(e,t,o,i){"use strict";function n(e,t,n){var r=Date.now(),s=o.__assign(o.__assign({},e),{msg:t,time:r-n}),a={group:"ajaxError",info:s};i.send(a,"except")}function r(e){return e>=200&&300>e}function s(e){if(!e||!e.entries())return"";for(var t=[],o=e.entries(),i=o.next();!i.done;i=o.next())t.push(i.value.join(": "));return t.join("\r\n")}function a(){XMLHttpRequest.prototype.open=function(){for(var e=[],t=0;t<arguments.length;t++)e[t]=arguments[t];var o={type:"xhr",method:e[0],url:e[1],msg:"",pageUrl:location.href};return $.extend(this,{xhrSpyInfo:o}),d.apply(this,e)},XMLHttpRequest.prototype.send=function(){for(var e=[],t=0;t<arguments.length;t++)e[t]=arguments[t];var o=this.xhrSpyInfo||{},i=Date.now(),s="string"==typeof e[0]?e[0]:"";return o.body=(s||"").slice(0,200),this.addEventListener("error",function(){(this.readyState!==XMLHttpRequest.DONE||r(this.status))&&n(o,"ajax network error",i)}),this.addEventListener("readystatechange",function(){this.readyState!==XMLHttpRequest.DONE||r(this.status)||n(o,"ajax error status-"+this.status,i)}),this.addEventListener("timeout",function(){n(o,"ajax timeout error",i)}),u.apply(this,e)}}function c(){var e=window.fetch;window.fetch=function(){for(var t=[],i=0;i<arguments.length;i++)t[i]=arguments[i];var r=t[0],a=t[1]||{body:"",method:"GET"},c="string"==typeof a.body?a.body:"",p=Date.now(),d={type:"fetch",method:a.method||"GET",url:r,body:(c||"").slice(0,200),msg:"",code:0,pageUrl:location.href};return e.apply(this,t).then(function(e){return e.ok||(d=o.__assign(o.__assign({},d),{response:s(e.headers),code:e.status}),e.clone().text().then(function(e){d.response=e}),n(d,"fetch error",p)),e})["catch"](function(e){var t=e.message||"fetch error";throw n(d,t,p),e})}}function p(){a(),c()}t.__esModule=!0,t.initInterceptor=t.initFetchInterceptor=t.initXHRInterceptor=t.fetchResponseHeadersToString=t.validateStatus=t.sendSpyLog=void 0;var d=XMLHttpRequest.prototype.open,u=XMLHttpRequest.prototype.send;t.sendSpyLog=n,t.validateStatus=r,t.fetchResponseHeadersToString=s,t.initXHRInterceptor=a,t.initFetchInterceptor=c,t.initInterceptor=p}),define("modules/page-log/ubc-log",["require","exports","tslib","@baidu/ubc-report-sdk/dist/ubc-web-sdk.umd.min"],function(e,t,o,i){"use strict";function n(e){var t=e+".init";i["default"](t,e,l)}function r(e){if("undefined"!=typeof window&&window.location&&window.location.href){var t=new URL(window.location.href);return t.searchParams.get(e)||""}return""}function s(e,t,s,a){void 0===a&&(a={}),n(e),i["default"](String(e)+".event",{type:t,value:s,ext:o.__assign({qid:bds.comm.qid||"",rsv_dl:r("rsv_dl")},a)})}function a(e,t,o){void 0===o&&(o={}),s(e,"show",t,o)}function c(e,t,o){void 0===o&&(o={}),s(e,"click",t,o)}function p(e,t){var o=document.getElementById("kw"),i=o instanceof HTMLInputElement?o.value:"";s(e,"click","search_kuang",{pos:t,query:i})}function d(e,t){var i=t.area||"",n=e.getAttribute("srcid")||"",r=e.getAttribute("id")||"",s=bds.comm.pageNum,a=e.clientHeight||0,c=bds.comm.pageSize||10,p=s?s*c-c:0,d=0;r&&s&&(d="RIGHT"===i?Number(r):+r-p);var u=e.firstElementChild||null,m=u?u.getAttribute("card-show-log"):null,l={};try{m&&(l=JSON.parse(m))}catch(g){l={}}var f=o.__assign(o.__assign({pn:p,srcid:n,page_num:s+"",height:a+"",abs_pos:r,relative_pos:d+""},t),l);return f}function u(e){var t=e.getAttribute("sub-show-log"),o=null;try{t&&(o=JSON.parse(t))}catch(i){return null}return o}function m(e,t,i){void 0===i&&(i="ala");var n={},r={},a=!1,c=[];t.forEach(function(e){e.card_extra.isSubEl?(r=u(e.el),n=d(e.card_extra.cardEl,e.card_extra),a=!0,r&&c.push(o.__assign(o.__assign({},n),{sub_valueinfo:o.__assign({},r),isSubEl:void 0,cardEl:void 0}))):(n=d(e.el,e.card_extra),c.push(n))}),(!a||c.length)&&s(e,"show",a?"subcard":"cardresult",{query:bds.comm.query||"",card_type:i,valueInfo:c})}t.__esModule=!0,t.sendCardLog=t.getCardConfig=t.sendIptLog=t.sendClickLog=t.sendShowLog=t.sendUbcLog=void 0,i=o.__importDefault(i);var l={version:"2.0",page:bds.comm.ishome?"home":"a",from:"search",queryParams:{appname:"pcsearch"},useSendBeacon:1};t.sendUbcLog=s,t.sendShowLog=a,t.sendClickLog=c,t.sendIptLog=p,t.getCardConfig=d,t.sendCardLog=m});var Cookie={set:function(e,t,o,i,n,r){document.cookie=e+"="+(r?t:escape(t))+(n?"; expires="+n.toGMTString():"")+(i?"; path="+i:"; path=/")+(o?"; domain="+o:"")},get:function(e,t){var o=document.cookie.match(new RegExp("(^| )"+e+"=([^;]*)(;|$)"));return null!=o?unescape(o[2]):t},clear:function(e,t,o){this.get(e)&&(document.cookie=e+"="+(t?"; path="+t:"; path=/")+(o?"; domain="+o:"")+";expires=Fri, 02-Jan-1970 00:00:00 GMT")}};!function(){function save(e){var t=[],o=[],i=[];for(tmpName in options)options.hasOwnProperty(tmpName)&&"duRobotState"!==tmpName&&(t.push('"'+tmpName+'":"'+options[tmpName]+'"'),o.push(tmpName),i.push(String(options[tmpName])));var n="{"+t.join(",")+"}";if(bds.comm.personalData)if(bds.comm.isNode){var r=$("#bsToken")&&$("#bsToken").val()||"";$.ajax({url:"/home/data/setups?bsToken="+r,type:"POST",data:JSON.stringify({props:o,values:i}),headers:{"Content-Type":"application/json"},traditional:!0,success:function(){writeCookie(),"function"==typeof e&&e()}})}else $.ajax({url:"/ups/submit/addtips?product=ps&from=pcfe&tips="+encodeURIComponent(n)+"&_r="+(new Date).getTime(),success:function(){writeCookie(),"function"==typeof e&&e()}});else writeCookie(),"function"==typeof e&&setTimeout(e,0)}function set(e,t){options[e]=t}function get(e){return options[e]}function writeCookie(){if(options.hasOwnProperty("sugSet")){var e="0"==options.sugSet?"0":"3";clearCookie("sug"),Cookie.set("sug",e,document.domain,"/",expire30y)}if(options.hasOwnProperty("sugStoreSet")){var e=0==options.sugStoreSet?"0":"1";clearCookie("sugstore"),Cookie.set("sugstore",e,document.domain,"/",expire30y)}var t=Cookie.get("BAIDUID"),o=t.match(/NR=(\d+)/)?t.match(/NR=(\d+)/)[1]:"";if(options.resultNum&&o!==options.resultNum.toString()&&writeBAIDUID(),(isHitSampling("close_predict_a")||isHitSampling("close_predict_b"))&&clearCookie("ORIGIN"),options.hasOwnProperty("isSwitch")){var i={0:"2",1:"0",2:"1"},e=i[options.isSwitch];clearCookie("ORIGIN"),isHitSampling("close_predict_a")||isHitSampling("close_predict_b")||Cookie.set("ORIGIN",e,document.domain,"/",expire30y)}if(options.hasOwnProperty("imeSwitch")){var e=options.imeSwitch;clearCookie("bdime"),Cookie.set("bdime",e,document.domain,"/",expire30y)}if(isHitSampling("PC_INDEX_BOX_REC_ON")&&options.hasOwnProperty("pcMainBoxRec")){var n=options.pcMainBoxRec;clearCookie("pcMainBoxRec"),Cookie.set("pcMainBoxRec",n,document.domain,"/",expire30y)}}function writeBAIDUID(){var e,t,o,i=Cookie.get("BAIDUID");/FG=(\d+)/.test(i)&&(t=RegExp.$1),/SL=(\d+)/.test(i)&&(o=RegExp.$1),/NR=(\d+)/.test(i)&&(e=RegExp.$1),options.hasOwnProperty("resultNum")&&(e=options.resultNum),options.hasOwnProperty("resultLang")&&(o=options.resultLang),Cookie.set("BAIDUID",i.replace(/:.*$/,"")+("undefined"!=typeof o?":SL="+o:"")+("undefined"!=typeof e?":NR="+e:"")+("undefined"!=typeof t?":FG="+t:""),".baidu.com","/",expire30y,!0)}function clearCookie(e){Cookie.clear(e,"/"),Cookie.clear(e,"/",document.domain),Cookie.clear(e,"/","."+document.domain),Cookie.clear(e,"/",".baidu.com")}function reset(e){options=defaultOptions,save(e)}var defaultOptions={sugSet:1,sugStoreSet:1,isSwitch:1,isJumpHttps:1,imeSwitch:0,resultNum:10,skinOpen:1,resultLang:0,duRobotState:"000"},options={},tmpName,expire30y=new Date;expire30y.setTime(expire30y.getTime()+94608e7);try{if(bds&&bds.comm&&bds.comm.personalData){if("string"==typeof bds.comm.personalData&&(bds.comm.personalData=eval("("+bds.comm.personalData+")")),!bds.comm.personalData)return;for(tmpName in bds.comm.personalData)defaultOptions.hasOwnProperty(tmpName)&&bds.comm.personalData.hasOwnProperty(tmpName)&&"SUCCESS"==bds.comm.personalData[tmpName].ErrMsg&&(options[tmpName]=bds.comm.personalData[tmpName].value)}try{parseInt(options.resultNum)||delete options.resultNum,parseInt(options.resultLang)||"0"==options.resultLang||delete options.resultLang}catch(e){}writeCookie(),"sugSet"in options||(options.sugSet=3!=Cookie.get("sug",3)?0:1),"sugStoreSet"in options||(options.sugStoreSet=Cookie.get("sugstore",0));var BAIDUID=Cookie.get("BAIDUID");"resultNum"in options||(options.resultNum=/NR=(\d+)/.test(BAIDUID)&&RegExp.$1?parseInt(RegExp.$1):10),"resultLang"in options||(options.resultLang=/SL=(\d+)/.test(BAIDUID)&&RegExp.$1?parseInt(RegExp.$1):0),"isSwitch"in options||(options.isSwitch=2==Cookie.get("ORIGIN",0)?0:1==Cookie.get("ORIGIN",0)?2:1),"imeSwitch"in options||(options.imeSwitch=Cookie.get("bdime",0)),!isHitSampling("PC_INDEX_BOX_REC_ON")||"pcMainBoxRec"in options||(options.pcMainBoxRec=Cookie.get("pcMainBoxRec",1))}catch(e){}window.UPS={writeBAIDUID:writeBAIDUID,reset:reset,get:get,set:set,save:save,writeCookie:writeCookie}}(),function(){require(["modules/monitor/js-except"],function(e){e.listenerExcept()}),require(["modules/ajax-log/ajax-log.service"],function(e){e.initInterceptor()});var e="https://pss.bdstatic.com/r/www/cache/static/protocol/https/plugins/every_cookie_4644b13.js";("Mac68K"==navigator.platform||"MacPPC"==navigator.platform||"Macintosh"==navigator.platform||"MacIntel"==navigator.platform)&&(e="https://pss.bdstatic.com/r/www/cache/static/protocol/https/plugins/every_cookie_mac_82990d4.js"),setTimeout(function(){$.ajax({url:e,cache:!0,dataType:"script"})},0);var t=navigator&&navigator.userAgent?navigator.userAgent:"",o=document&&document.cookie?document.cookie:"",i=!!(t.match(/(msie [2-8])/i)||t.match(/windows.*safari/i)&&!t.match(/chrome/i)||t.match(/(linux.*firefox)/i)||t.match(/Chrome\/29/i)||t.match(/mac os x.*firefox/i)||o.match(/\bISSW=1/)||0==UPS.get("isSwitch"));bds&&bds.comm&&(bds.comm.supportis=isHitSampling("close_predict_a")||isHitSampling("close_predict_b")?!1:!i,bds.comm.isui=!0),window.__restart_confirm_timeout=!0,window.__confirm_timeout=8e3,window.__disable_is_guide=!0,window.__disable_swap_to_empty=!0,window.__switch_add_mask=!0,bds.comm.newindex&&$(window).on("index_off",function(){$('<div class="c-tips-container" id="c-tips-container"></div>').insertAfter("#wrapper"),window.__sample_dynamic_tab&&$("#s_tab").remove()}),bds.comm&&bds.comm.ishome&&Cookie.get("H_PS_PSSID")&&(bds.comm.indexSid=Cookie.get("H_PS_PSSID"));var n=$(document).find("#s_tab").find("a");n&&n.length>0&&n.each(function(e,t){t.innerHTML&&t.innerHTML.match(/新闻/)&&(t.innerHTML="资讯",t.href="//www.baidu.com/s?rtt=1&bsst=1&cl=2&tn=news&word=",t.setAttribute("sync",!0))})}();</script><script type="text/javascript" src="https://pss.bdstatic.com/r/www/cache/static/protocol/https/bundles/es6-polyfill_388d059.js"></script><script type="text/javascript" src="https://pss.bdstatic.com/r/www/cache/static/protocol/https/bundles/polyfill_9354efa.js"></script><script type="text/javascript" src="https://pss.bdstatic.com/r/www/cache/static/protocol/https/global/js/all_async_search_2c0d868.js"></script><script>
    if(bds.comm.supportis){
        window.__restart_confirm_timeout=true;
        window.__confirm_timeout=8000;
        window.__disable_is_guide=true;
        window.__disable_swap_to_empty=true;
    }

    if(typeof initPreload == "function"){
        initPreload({
            'isui':true,
            'index_form':"#form",
            'index_kw':"#kw",
            'result_form':"#form",
            'result_kw':"#kw",
            'isui':true
        });
    }
    else{
        window._sp_async = undefined;
        new Image().src = "/home/page/data/pageserver?errno=7004&from=superman&_t" + new Date() * 1;
    }

</script><script type="text/javascript">(function(){var index_head_css=$("#style_super_head_inline");var index_is_css=$("#s_is_index_css");window.index_links=[];var head_index_css=$("head [index]");window.index_on=function(){s_session.index_off=false};var resultLogo="//www.baidu.com/img/flexible/logo/pc/result.png";var resultNewLogo="//www.baidu.com/img/flexible/logo/pc/result@2.png";var resultPeakLogo="//www.baidu.com/img/flexible/logo/pc/peak-result.png";var resultGray="//www.baidu.com/img/flexible/logo/pc/result_gray.png"
;var changeResultLogo=function(src,newSrc,peakSrc){if($("#result_logo .index-logo-src").attr("src")!==src){$("#result_logo .index-logo-src").attr("src",src);$("#result_logo .index-logo-srcnew").attr("src",newSrc);$("#result_logo .index-logo-peak").attr("src",peakSrc)}};window.index_off=function(){var sFormWrapper=$(".s_form .s_form_wrapper");if(bds.comm.inputColorConfig&&bds.comm.inputColorConfig.barColor&&bds.comm.sIndex){sFormWrapper.removeClass("scene-style")}s_session.index_off=true
;bds.comm.sIndex=0;$(document.body).attr("link","#0000cc");index_links=[];$("head").find("link").each(function(){if($(this).attr("data-for")!=="result"){index_links.push(this);$(this).remove()}});index_is_css.remove();index_head_css.remove();$("#head_wrapper").css({width:"",marginBottom:0});$("#kw").css("font-family","");if(bds&&bds.comm&&bds.comm.boxRecSwitch==="1"){document.getElementById("kw").placeholder=""}var decodeHTML=function(source){var str=String(source).replace(/&quot;/g,'"'
).replace(/&lt;/g,"<").replace(/&gt;/g,">").replace(/&amp;/g,"&");return str.replace(/&#([\d]+);/g,function(_0,_1){return String.fromCharCode(parseInt(_1,10))})};$("head").append(decodeHTML($("#s_index_off_css").html()));$("head").append(decodeHTML($("#s_is_result_css").html()));$(".s-tips-skin").remove();$(".s-skin-container").remove();$("#s_mancard_newmusic").remove();var wrapper=$("#wrapper");if(wrapper.hasClass("new_input_superman")){wrapper.addClass("new_input_wrapper");wrapper.removeClass(
"new_input_superman")}var body=$("body");var pcFreshWrapper=bds.comm.pcFreshWrapper;var pcFreshTitle=bds.comm.pcFreshTitle;var pcFreshCard=bds.comm.pcFreshCard;var pcFreshSmooth=bds.comm.pcFreshSmooth;if(pcFreshWrapper){body.addClass("pc-fresh-wrapper");switch(pcFreshWrapper){case"PC_FRESH_WRAPPER_EXTREME":body.addClass("pc-fresh-wrapper-ext");break;case"PC_FRESH_WRAPPER_CONSERVATIVE":body.addClass("pc-fresh-wrapper-con");break}}else{var styleContent=decodeHTML(
'<style type="text/css">@media screen and (min-width: 1116px) { html { overflow-y: auto; overflow-x: hidden; } }</style>');$("head").append(styleContent)}pcFreshCard&&body.addClass("pc-fresh-card");if(pcFreshTitle){switch(pcFreshTitle){case"PC_FRESH_TITLE_EXTREME":body.addClass("pc-fresh-title-ext");break;case"PC_FRESH_TITLE_CONSERVATIVE":body.addClass("pc-fresh-title-con");break}}pcFreshSmooth&&body.addClass("pc-fresh-smooth");head_index_css.remove();$("#s-hotsearch-wrapper").remove();$(
"#s_side_wrapper").remove();$("#blind-box").remove();$("#s_popup_advert").remove();$("#video-meeting").remove();var isPreSearchboxGray=bds.comm.isPreSearchboxGray||"0";var $body=$("body");if(isPreSearchboxGray==="0"){$body.hasClass("big-event-gray")&&$("body").removeClass("big-event-gray");changeResultLogo(resultLogo,resultNewLogo,resultPeakLogo)}if(isPreSearchboxGray==="1"&&!$body.hasClass("big-event-gray")){$("body").addClass("big-event-gray");changeResultLogo(resultGray,resultGray,resultGray)
}var nodeSample=bds&&bds.comm&&bds.comm.nodeSample||"";var boxRec=nodeSample.indexOf("need_carousels")>-1||nodeSample.indexOf("no_need_carousels")>-1;if(boxRec){$(".placeholder").remove()}}})();</script><!--[if < IE 9]><script type="text/javascript" src="https://pss.bdstatic.com/static/superman/js/polyfill-ie8-30f98ab294.js"></script><![endif]--><script type="text/javascript" src="https://pss.bdstatic.com/static/superman/js/sbase-97ef1516e5.js"></script><link href="https://pss.bdstatic.com/static/superman/css/ubase_sync-d600f57804.css?v=md5" type="text/css" rel="stylesheet" InlineContent /><script>F._setMod("superman");F._fileMap({"/js/sbase-97ef1516e5.js":["lib/event.ts","lib/extract_data.ts","lib/commonUtils.ts","plugin/browser","plugin/strpx","plugin/url","plugin/ajax","plugin/load_file","plugin/string","plugin/stringify","plugin/localstorage","plugin/fn","plugin/easing","plugin/mousewheel","plugin/cookie","lib/base","lib/sbase","lib/class","lib/mod_evt","lib/log","lib/thunder","lib/utils.js","common/user_attr","common/select","common/bottom_layer","components/ubc.ts",
"components/initSan.ts"],"js/polyfill-ie8-30f98ab294.js":["polyfill/object-define-property","polyfill/object-keys"],"/js/min_super-c25d576de3.js":["common/result_page","common/image_lazy_load","log/webb.min","log/super_all","lib/carditem_log","ps/log","ps/sindex","ps/sug","ps/fpid","start/super_start","page/info","page/scroll"],"/js/super_ext-a0b60bd05d.js":["plugin/mousewheel","weather/setting_refresh","weather/setting_ctrl","weather/setting_view","weather/weather_city"],
"/js/resource_error_monitor-ed61c2f0be.js":["page/resource_error_monitor.ts"],"/js/super_load-b41edfc5ce.js":["weather/weather_tpl","weather/weather_ctrl","weather/weather_autorefresh","start/skin_start","skin/skin_rewrite","mngr/top_layer","mngr/menu_user","mngr/menu_common","mngr/quit_dialog","mngr/top_menunav_new","common/guide_tip","page/page_exp"],"/js/skin_layer-72c77f7044.js":["skin/skin_init_new","skin/skin_cut_img","skin/skin_model","skin/skin_nav","skin/skin_page","skin/skin_tools",
"skin/skin_img","skin/skin_control_new","skin/skin_view","skin/skin_new_upload","skin/skin_setting","skin/skin_preview","skin/skin_defined","skin/skin_opacity","skin/skin_ajax","skin/skin_extra","skin/skin_random"],"/js/opacity_mod-57b92b76b5.js":["opacity/opacity_tpl","opacity/opacity_action","opacity/opacity_conf"],"/js/min_setting-c3f73c1e26.js":["setting/setting_constructor","setting/mod_drag","setting/setting_action"],"/js/page_setting-f890166cc6.js":["log/super_all","setting/page_setting",
"setting/tab_msg","start/page_setting_start","common/pop","setting/yaohao","msg/const","setting/data_adapter","msg/bind","msg/bind_window","setting/common_setting"],"/js/mancard-d862391a9d.js":["skeleton/card","skeleton/model","skeleton/card_collection","skeleton/card_showLog","skeleton/card_alsLog","skeleton/presenter","skeleton/card_tab","skeleton/card_refresh","skeleton/toast","setting/card_setting","subscribe/parabola","subscribe/subscribe"],"/js/subscribe-d2630c5b22.js":[
"subscribe/subscribe"],"/js/s_super_index-895c0c52f8.js":["ps/autohover_input","page/ua_monitor"],"/js/s_super_async-80d96816da.js":["node_modules/@searchfe/inject-js/dist/amd/index.js","node_modules/@searchfe/reflect-metadata/Reflect.js","ps/autohover_input","components/drag/dragComponent","components/drag/dragEmitter","components/drag/dragTool","components/drag/draggable","components/drag/dragsort","components/drag/dragdir"],"/js/min_mt-3df544d11c.js":["msg/const","msg/data_server",
"msg/data_adapter"],"/js/components/content-info-12dbf9fb6d.js":["components/content-info.ts"],"/js/components/csaitab-log-a9c9cae804.js":["components/csaitab-log.ts"],"/js/components/top-right-operate/operate-b7693b0cc7.js":["components/top-right-operate/operate.ts"],"/js/components/new-search-guide-bub-2ac19f2971.js":["components/new-search-guide-bub.ts"],"/js/mt_ext-4eee81afe1.js":["msg/index_nav_msg_extend","msg/msg_builder","msg/bind","msg/bind_window"],"/js/min_notice-605fc17cf9.js":[
"mt/mt_show","mt/mt_msg_constructor"],"/js/notice_ext-0bfec0da78.js":["mt/mt_ext","mt/mt_msg_item","mt/mt_msg_operate"],"/js/ps/ishttps-8b74cccb13.js":["ps/ishttps"],"/js/ps/async-destroy-b00b99a55b.js":["ps/async-destroy"],"/js/lib/exception-13bf584742.js":["lib/exception"],"/js/lib/jquery-1-edb203c114.10.2.js":["lib/jquery-1.10.2"],"/js/lib/esl-d776bfb1aa.js":["lib/esl"],"/js/ps/yc-4c3e9b9cb3.js":["ps/yc"],"/js/components/hotsearch-7a667d2dda.js":["components/hotsearch"],
"/js/components/placeholder-b83e2493f9.js":["components/placeholder.ts"],"/js/components/qrcode-0e4b67354f.js":["components/qrcode"],"/js/components/advert-064271ed9b.js":["components/advert"],"/js/components/nav-d9eab6af09.js":["components/nav"],"/js/components/tips-e2ceadd14d.js":["components/tips"],"/js/components/guide-8759cd328f.js":["components/guide"],"/js/components/bbox/bbox-view-df7e8cb3b0.js":["components/bbox/bbox_view.ts","components/bbox/bbox_animate.ts"],
"/js/components/guide_tips-d9e617f782.js":["components/guide_tips.ts"],"/js/components/login_guide-4fba3971ce.js":["components/login_guide.ts"],"/js/components/video-meet-7833028d86.js":["components/video-meet.ts"],"/amd_modules/@baidu/video-meeting-1be7f62dac.js":["@baidu/video-meeting/dist/index","@baidu/video-meeting"],"/amd_modules/@baidu/aging-tools-pc-1e5afe8bdf.js":["@baidu/aging-tools-pc/dist/index","@baidu/aging-tools-pc"],"/amd_modules/@baidu/ai-search-box-entry-ea20fec552.js":[
"@baidu/ai-search-box-entry/dist/js/index"],"/js/components/aging-tools-094cd28890.js":["components/aging-tools.ts"],"/amd_modules/tslib-c95383af0c.js":["tslib","./tslib/tslib"],"/js/components/invoke-97e9694cb9.js":["components/invoke.ts"],"/js/news_new/index-117b80ef84.js":["news_new/init_tts","news_new/card_alsLog","news_new/ck","news_new/index_view","news_new/index"],"/js/min_tips-3c72b5b27b.js?v=md5":["log/log","start/tips_start"],"/js/ubase-dddde7cd4e.js?v=md5":["superuijs/superui",
"superuijs/util/tool","superuijs/util/emitter","superuijs/util/dot","superuijs/util/smDot","superuijs/component","superuijs/component/draggable","superuijs/component/scrollbar","superuijs/component/dialog","superuijs/component/tips","superuijs/component/share","superuijs/component/share2","superuijs/component/notify","superuijs/component/suggestion","superuijs/component/placeholder"],"/js/ubase_unused-e357fff740.js?v=md5":["superuijs/component/draggsort","superuijs/component/draggselect",
"superuijs/component/draggdirs"],"/js/utils-ce7201f913.js?v=md5":["superuijs/util/pubsub"],"/css/super_ext-36b360db2f.css":["mngr_quit_guid.css"],"/css/components/user_quit_dialog-527f3ede74.css":["components/user_quit_dialog.css"],"/css/super_min-6f52ab3317.css":["super_init.css","skin.css","super_skin.css","select.css","index_guide.css"],"/css/skin/skin_layer-d2120713ac.css":["skin/skin_layer.css"],"/css/guidetip-b41ed60c8a.css":["guide.css"],"/css/nsguide-a66438b784.css":["ns_guide.css"],
"/css/setting_min-4dae12391e.css":["setting.css"],"/css/card_setting-673b2943c1.css":["card_setting.css"],"/css/page_setting-da9f239ab9.css":["super_setting.css","select.css","setting_msg.css","setting_yaohao.css","pop.css","dialog.css"],"/css/mt_min-88763bc1c9.css":["mt_mod.css","index_msg.css"],"/css/pages/index-997a76c5f7.css":["pages/index.css","components/qrcode_new.css"],"/css/pages/index_result-3a33835e40.css":["pages/index_result.css"],"/css/pages/result-98fff35cad.css":[
"pages/result.css"],"/css/pages/ps-38785f7e87.css":["pages/ps.css","grid/grid.css"],"/css/weather/weather_new-4aa68cfe53.css":["weather/weather_new.css","weather/weather_setting_new.css"],"/css/pages/common-d7acedee34.css":["components/head_wrapper_new.css","components/recommend_search_list.css","s_top_wrap_new.css","components/topmenu_new.css","bottom_layer.css","components/blind_box.css","popup_advert.css","components/guide_info.css","components/aging_tool.css","components/gray.css",
"components/drag.css","components/new_search_guide_bub.css"],"/css/components/hotsearch-27c3aacbe4.css":["components/hotsearch.css"],"/css/components/nav-ffd5baab45.css":["components/nav.css"],"/css/card_min2-ed227fefbd.css":["card_menu2.css","card_side2.css"],"/css/recommand/init-c52228535a.css":["news_new/init.css","news_new/waterfall.css","news_new/custom_sui.css","news_new/hot_search.css","news_new/dialog.css"],"/css/ubase_sync-d600f57804.css?v=md5":["scrollbar_sync.css"],
"/css/ubase-89d6b96e41.css?v=md5":["superui/superui.css","superui/dialog.css","superui/tips.css","superui/share.css","superui/scrollbar.css","superui/suggestion.css"],"/css/ubase_sync-d600f57804.css?v=md5":["superui/scrollbar_sync.css"]});window._xman_speed=window._xman_speed||{};F._setContext({base:"lib/sbase"});F.use("lib/mod_evt",function(evt){F._setContextMethod("fire",function(evtName,evtArgs){return evt.fire(this.svnMod+":"+this.modName,evtName,evtArgs)});F._setContextMethod("listen",
function(modName,evtName,handler){var mType=Object.prototype.toString.call(modName);if(mType=="[object String]"){if(modName.indexOf(":")<0){modName=this.svnMod+":"+modName}}else if(mType=="[object Array]"){for(var i=0,len=modName.length;i<len;i++){if(modName[i].indexOf(":")<0){modName[i]=this.svnMod+":"+modName[i]}}}evt.on(modName,evtName,handler)});F._setContextMethod("unListen",function(modName,evtName,handler){evt.un(this.svnMod+":"+this.modName,evtName,handler)})});F._loadScriptTimeout=15e3
;F._useConfig=true;F._firstScreenCSS=F._firstScreenCSS||[];F._firstScreenCSS.push("/css/super_min-6f52ab3317.css");F._firstScreenJS=F._firstScreenJS||[];F._firstScreenJS.push("/js/min_super-c25d576de3.js");</script><script>
$(window).on("load", function() {
    var e = $('#virus-2020');
    e.click(function() {
        $.setCookie('virus-2020-clicked', '1');
        e.removeClass('dot');
    });
    var hasClicked = $.getCookie && $.getCookie('virus-2020-clicked');
    if (!hasClicked) {
        e.addClass('dot');
    }
});
</script><script>
if (typeof F === 'object') {
    F.use('superman:skeleton/presenter', function (presenter) {
        presenter.desktopEvent();
    });
}
</script><script>(function(){var amd={keys:{},addPaths:function(opt){if(typeof opt!=="object"){return}for(var key in opt){if(opt.hasOwnProperty(key)){opt[key]=opt[key].replace(/(\.js)/,"")}}require.config({paths:opt});$.extend(amd.keys,opt)},addConfig:function(config){if(typeof config!=="object"){return}require.config(config);for(var item in config){if(config.hasOwnProperty(item)){if(!amd[item]){amd[item]={}}for(var key in config[item]){if(config[item].hasOwnProperty(key)&&config[item][key]){
amd[item][key]=config[item][key]}}}}}};s_domain.amd=amd;var staticUrl=s_domain.staticUrl+"static/superman/";amd.addPaths({"@baidu/video-meeting":staticUrl+"amd_modules/@baidu/video-meeting-1be7f62dac.js","superman/components/video-meet":staticUrl+"js/components/video-meet-7833028d86.js",tslib:staticUrl+"amd_modules/tslib-c95383af0c.js","superman/components/guide_tips":staticUrl+"js/components/guide_tips-d9e617f782.js",
"superman/components/login_guide":staticUrl+"js/components/login_guide-4fba3971ce.js","superman/components/bbox/bbox_view":staticUrl+"js/components/bbox/bbox-view-df7e8cb3b0.js","@baidu/aging-tools-pc":staticUrl+"amd_modules/@baidu/aging-tools-pc-1e5afe8bdf.js","superman/components/aging-tools":staticUrl+"js/components/aging-tools-094cd28890.js","superman/components/ubc":staticUrl+"js/components/ubc.js",
"superman/components/top-right-operate/operate":staticUrl+"js/components/top-right-operate/operate-b7693b0cc7.js","superman/components/content-info":staticUrl+"js/components/content-info-12dbf9fb6d.js","superman/components/csaitab-log":staticUrl+"js/components/csaitab-log-a9c9cae804.js","superman/components/invoke":staticUrl+"js/components/invoke-97e9694cb9.js","superman/components/placeholder":staticUrl+"js/components/placeholder-b83e2493f9.js",
"superman/components/new-search-guide-bub":staticUrl+"js/components/new-search-guide-bub-2ac19f2971.js","ai-search-box-entry":staticUrl+"/amd_modules/@baidu/ai-search-box-entry-ea20fec552.js"})})();</script><script>F._setMod("superman");F._fileMap({"/js/sbase-97ef1516e5.js":["lib/event.ts","lib/extract_data.ts","lib/commonUtils.ts","plugin/browser","plugin/strpx","plugin/url","plugin/ajax","plugin/load_file","plugin/string","plugin/stringify","plugin/localstorage","plugin/fn","plugin/easing","plugin/mousewheel","plugin/cookie","lib/base","lib/sbase","lib/class","lib/mod_evt","lib/log","lib/thunder","lib/utils.js","common/user_attr","common/select","common/bottom_layer","components/ubc.ts",
"components/initSan.ts"],"js/polyfill-ie8-30f98ab294.js":["polyfill/object-define-property","polyfill/object-keys"],"/js/min_super-c25d576de3.js":["common/result_page","common/image_lazy_load","log/webb.min","log/super_all","lib/carditem_log","ps/log","ps/sindex","ps/sug","ps/fpid","start/super_start","page/info","page/scroll"],"/js/super_ext-a0b60bd05d.js":["plugin/mousewheel","weather/setting_refresh","weather/setting_ctrl","weather/setting_view","weather/weather_city"],
"/js/resource_error_monitor-ed61c2f0be.js":["page/resource_error_monitor.ts"],"/js/super_load-b41edfc5ce.js":["weather/weather_tpl","weather/weather_ctrl","weather/weather_autorefresh","start/skin_start","skin/skin_rewrite","mngr/top_layer","mngr/menu_user","mngr/menu_common","mngr/quit_dialog","mngr/top_menunav_new","common/guide_tip","page/page_exp"],"/js/skin_layer-72c77f7044.js":["skin/skin_init_new","skin/skin_cut_img","skin/skin_model","skin/skin_nav","skin/skin_page","skin/skin_tools",
"skin/skin_img","skin/skin_control_new","skin/skin_view","skin/skin_new_upload","skin/skin_setting","skin/skin_preview","skin/skin_defined","skin/skin_opacity","skin/skin_ajax","skin/skin_extra","skin/skin_random"],"/js/opacity_mod-57b92b76b5.js":["opacity/opacity_tpl","opacity/opacity_action","opacity/opacity_conf"],"/js/min_setting-c3f73c1e26.js":["setting/setting_constructor","setting/mod_drag","setting/setting_action"],"/js/page_setting-f890166cc6.js":["log/super_all","setting/page_setting",
"setting/tab_msg","start/page_setting_start","common/pop","setting/yaohao","msg/const","setting/data_adapter","msg/bind","msg/bind_window","setting/common_setting"],"/js/mancard-d862391a9d.js":["skeleton/card","skeleton/model","skeleton/card_collection","skeleton/card_showLog","skeleton/card_alsLog","skeleton/presenter","skeleton/card_tab","skeleton/card_refresh","skeleton/toast","setting/card_setting","subscribe/parabola","subscribe/subscribe"],"/js/subscribe-d2630c5b22.js":[
"subscribe/subscribe"],"/js/s_super_index-895c0c52f8.js":["ps/autohover_input","page/ua_monitor"],"/js/s_super_async-80d96816da.js":["node_modules/@searchfe/inject-js/dist/amd/index.js","node_modules/@searchfe/reflect-metadata/Reflect.js","ps/autohover_input","components/drag/dragComponent","components/drag/dragEmitter","components/drag/dragTool","components/drag/draggable","components/drag/dragsort","components/drag/dragdir"],"/js/min_mt-3df544d11c.js":["msg/const","msg/data_server",
"msg/data_adapter"],"/js/components/content-info-12dbf9fb6d.js":["components/content-info.ts"],"/js/components/csaitab-log-a9c9cae804.js":["components/csaitab-log.ts"],"/js/components/top-right-operate/operate-b7693b0cc7.js":["components/top-right-operate/operate.ts"],"/js/components/new-search-guide-bub-2ac19f2971.js":["components/new-search-guide-bub.ts"],"/js/mt_ext-4eee81afe1.js":["msg/index_nav_msg_extend","msg/msg_builder","msg/bind","msg/bind_window"],"/js/min_notice-605fc17cf9.js":[
"mt/mt_show","mt/mt_msg_constructor"],"/js/notice_ext-0bfec0da78.js":["mt/mt_ext","mt/mt_msg_item","mt/mt_msg_operate"],"/js/ps/ishttps-8b74cccb13.js":["ps/ishttps"],"/js/ps/async-destroy-b00b99a55b.js":["ps/async-destroy"],"/js/lib/exception-13bf584742.js":["lib/exception"],"/js/lib/jquery-1-edb203c114.10.2.js":["lib/jquery-1.10.2"],"/js/lib/esl-d776bfb1aa.js":["lib/esl"],"/js/ps/yc-4c3e9b9cb3.js":["ps/yc"],"/js/components/hotsearch-7a667d2dda.js":["components/hotsearch"],
"/js/components/placeholder-b83e2493f9.js":["components/placeholder.ts"],"/js/components/qrcode-0e4b67354f.js":["components/qrcode"],"/js/components/advert-064271ed9b.js":["components/advert"],"/js/components/nav-d9eab6af09.js":["components/nav"],"/js/components/tips-e2ceadd14d.js":["components/tips"],"/js/components/guide-8759cd328f.js":["components/guide"],"/js/components/bbox/bbox-view-df7e8cb3b0.js":["components/bbox/bbox_view.ts","components/bbox/bbox_animate.ts"],
"/js/components/guide_tips-d9e617f782.js":["components/guide_tips.ts"],"/js/components/login_guide-4fba3971ce.js":["components/login_guide.ts"],"/js/components/video-meet-7833028d86.js":["components/video-meet.ts"],"/amd_modules/@baidu/video-meeting-1be7f62dac.js":["@baidu/video-meeting/dist/index","@baidu/video-meeting"],"/amd_modules/@baidu/aging-tools-pc-1e5afe8bdf.js":["@baidu/aging-tools-pc/dist/index","@baidu/aging-tools-pc"],"/amd_modules/@baidu/ai-search-box-entry-ea20fec552.js":[
"@baidu/ai-search-box-entry/dist/js/index"],"/js/components/aging-tools-094cd28890.js":["components/aging-tools.ts"],"/amd_modules/tslib-c95383af0c.js":["tslib","./tslib/tslib"],"/js/components/invoke-97e9694cb9.js":["components/invoke.ts"],"/js/news_new/index-117b80ef84.js":["news_new/init_tts","news_new/card_alsLog","news_new/ck","news_new/index_view","news_new/index"],"/js/min_tips-3c72b5b27b.js?v=md5":["log/log","start/tips_start"],"/js/ubase-dddde7cd4e.js?v=md5":["superuijs/superui",
"superuijs/util/tool","superuijs/util/emitter","superuijs/util/dot","superuijs/util/smDot","superuijs/component","superuijs/component/draggable","superuijs/component/scrollbar","superuijs/component/dialog","superuijs/component/tips","superuijs/component/share","superuijs/component/share2","superuijs/component/notify","superuijs/component/suggestion","superuijs/component/placeholder"],"/js/ubase_unused-e357fff740.js?v=md5":["superuijs/component/draggsort","superuijs/component/draggselect",
"superuijs/component/draggdirs"],"/js/utils-ce7201f913.js?v=md5":["superuijs/util/pubsub"],"/css/super_ext-36b360db2f.css":["mngr_quit_guid.css"],"/css/components/user_quit_dialog-527f3ede74.css":["components/user_quit_dialog.css"],"/css/super_min-6f52ab3317.css":["super_init.css","skin.css","super_skin.css","select.css","index_guide.css"],"/css/skin/skin_layer-d2120713ac.css":["skin/skin_layer.css"],"/css/guidetip-b41ed60c8a.css":["guide.css"],"/css/nsguide-a66438b784.css":["ns_guide.css"],
"/css/setting_min-4dae12391e.css":["setting.css"],"/css/card_setting-673b2943c1.css":["card_setting.css"],"/css/page_setting-da9f239ab9.css":["super_setting.css","select.css","setting_msg.css","setting_yaohao.css","pop.css","dialog.css"],"/css/mt_min-88763bc1c9.css":["mt_mod.css","index_msg.css"],"/css/pages/index-997a76c5f7.css":["pages/index.css","components/qrcode_new.css"],"/css/pages/index_result-3a33835e40.css":["pages/index_result.css"],"/css/pages/result-98fff35cad.css":[
"pages/result.css"],"/css/pages/ps-38785f7e87.css":["pages/ps.css","grid/grid.css"],"/css/weather/weather_new-4aa68cfe53.css":["weather/weather_new.css","weather/weather_setting_new.css"],"/css/pages/common-d7acedee34.css":["components/head_wrapper_new.css","components/recommend_search_list.css","s_top_wrap_new.css","components/topmenu_new.css","bottom_layer.css","components/blind_box.css","popup_advert.css","components/guide_info.css","components/aging_tool.css","components/gray.css",
"components/drag.css","components/new_search_guide_bub.css"],"/css/components/hotsearch-27c3aacbe4.css":["components/hotsearch.css"],"/css/components/nav-ffd5baab45.css":["components/nav.css"],"/css/card_min2-ed227fefbd.css":["card_menu2.css","card_side2.css"],"/css/recommand/init-c52228535a.css":["news_new/init.css","news_new/waterfall.css","news_new/custom_sui.css","news_new/hot_search.css","news_new/dialog.css"],"/css/ubase_sync-d600f57804.css?v=md5":["scrollbar_sync.css"],
"/css/ubase-89d6b96e41.css?v=md5":["superui/superui.css","superui/dialog.css","superui/tips.css","superui/share.css","superui/scrollbar.css","superui/suggestion.css"],"/css/ubase_sync-d600f57804.css?v=md5":["superui/scrollbar_sync.css"]});window._xman_speed=window._xman_speed||{};F._setContext({base:"lib/sbase"});F.use("lib/mod_evt",function(evt){F._setContextMethod("fire",function(evtName,evtArgs){return evt.fire(this.svnMod+":"+this.modName,evtName,evtArgs)});F._setContextMethod("listen",
function(modName,evtName,handler){var mType=Object.prototype.toString.call(modName);if(mType=="[object String]"){if(modName.indexOf(":")<0){modName=this.svnMod+":"+modName}}else if(mType=="[object Array]"){for(var i=0,len=modName.length;i<len;i++){if(modName[i].indexOf(":")<0){modName[i]=this.svnMod+":"+modName[i]}}}evt.on(modName,evtName,handler)});F._setContextMethod("unListen",function(modName,evtName,handler){evt.un(this.svnMod+":"+this.modName,evtName,handler)})});F._loadScriptTimeout=15e3
;F._useConfig=true;F._firstScreenCSS=F._firstScreenCSS||[];F._firstScreenCSS.push("/css/super_min-6f52ab3317.css");F._firstScreenJS=F._firstScreenJS||[];F._firstScreenJS.push("/js/min_super-c25d576de3.js");</script><script id="s_js_mancard_main" data-src="https://pss.bdstatic.com/static/superman/js/mancard-d862391a9d.js">
    _manCard.asynLoad("s_js_mancard_main");
</script><script id="s_js_mancard_subscribe" data-src="https://pss.bdstatic.com/static/superman/js/subscribe-d2630c5b22.js">
        _manCard.asynLoad("s_js_mancard_subscribe");
    </script><script src="https://pss.bdstatic.com/static/superman/js/components/hotsearch-7a667d2dda.js"></script><script type="text/javascript" src="https://pss.bdstatic.com/static/superman/js/min_super-c25d576de3.js"></script><script>
(function(){
    var skinOff = "\/\/www.baidu.com\/img\/PCtm_d9c8750bed0b3c7d089fa7d55720d6cf.png",
        skinOffRe = "\/\/www.baidu.com\/img\/PCtm_d9c8750bed0b3c7d089fa7d55720d6cf.png",
        skinOn = "\/\/www.baidu.com\/img\/PCfb_5bf082d29588c07f842ccde3f97243ea.png",
        skinOnRe = "\/\/www.baidu.com\/img\/PCfb_5bf082d29588c07f842ccde3f97243ea.png",
        userSetLogoWrite = "https:\/\/pss.bdstatic.com\/static\/superman\/img\/logo\/logo_white-d0c9fe2af5.png",
        userSetLogoRed = "https:\/\/pss.bdstatic.com\/static\/superman\/img\/logo\/logo_redBlue-0a7c20fcaa.png";
        
    var greatEvent = {};
    var useSkin = "off";
    
    if (greatEvent && greatEvent.switch === '1' && useSkin !== 'on') {
        if (greatEvent.doodleExist === '0' || (greatEvent.doodleExist === '1' && greatEvent.doodleCover === '1')) {
            var skinOff = '//www.baidu.com/img/flexible/logo/pc/index_gray.png';
            var skinOffRe = '//www.baidu.com/img/flexible/logo/pc/index_gray.png';
        }
    }
    

    if(!skinOn) {
        skinOn = skinOff;
        skinOnRe = skinOffRe;
    }
    var change = function(src, newsrc){
        if ($('#s_lg_img').attr('src')!=src){
            $('#s_lg_img').attr('src',src);
            $('#s_lg_img_new').attr('src',newsrc);
        }
    }

    var changeResultLogo = function (src, newSrc, peakSrc) {
        if ($('#result_logo .index-logo-src').attr('src') !== src) {
            $('#result_logo .index-logo-src').attr('src', src);
            $('#result_logo .index-logo-srcnew').attr('src', newSrc);
            $('#result_logo .index-logo-peak').attr('src', peakSrc);
        }
    }
    if (s_session.userSkinName === '') {
        change(skinOff, skinOffRe);
    }
    $(window).on('s-skinoff', function () {
        change(skinOff, skinOffRe);
        if (greatEvent && greatEvent.switch === '1' && !$('body').hasClass('big-event-gray')) {
            $('body').addClass('big-event-gray');
            if (greatEvent.doodleExist === '0' || (greatEvent.doodleExist === '1' && greatEvent.doodleCover === '1')) {
                var hbLogo = "\/\/www.baidu.com\/img\/flexible\/logo\/pc\/index_gray.png";
                change(hbLogo, hbLogo);
            }
            var hbResultLogo = "\/\/www.baidu.com\/img\/flexible\/logo\/pc\/result_gray.png";
            changeResultLogo(hbResultLogo, hbResultLogo, hbResultLogo);
        }
    });

    var resultLogo = '//www.baidu.com/img/flexible/logo/pc/result.png';
    var resultNewLogo = '//www.baidu.com/img/flexible/logo/pc/result@2.png';
    var resultPeakLogo = '//www.baidu.com/img/flexible/logo/pc/peak-result.png';

    $(window).on('s-skinon', function () {
        var $skinContainer = $('.s-skin-container');
        // 如果有置灰类名，且用户开启自定义皮肤，去除相关置灰类名
        if ($('body').hasClass('big-event-gray') && useSkin === 'on') {
            $('body').removeClass('big-event-gray');
            $skinContainer.length > 0 && $skinContainer.hasClass('skin-gray-event') && $skinContainer.removeClass('skin-gray-event');
        }
        // 如果开启置灰，且用户未使用自定义皮肤，需要将皮肤置灰
        if (greatEvent && greatEvent.switch === '1' && useSkin !== 'on') {
            $skinContainer.length > 0 && !$skinContainer.hasClass('skin-gray-event') && $skinContainer.addClass('skin-gray-event');
        }
        var LogoColor = $('#head');
        if (LogoColor.hasClass('red-blue-logo')) {
            if (s_session.logoCode) {
                change(skinOn, skinOnRe);
            } else {
                change(userSetLogoRed, userSetLogoRed);
            }
        } else if (LogoColor.hasClass('white-logo')){
                var isUpload = s_session.userLogoIsupload === 'false' ? false : s_session.userLogoIsupload;
                if (!isUpload || s_session.logoCode) {
                    change(skinOn, skinOnRe);
                } else {
                    change(userSetLogoWrite, userSetLogoWrite);
                }
        } else {
            change(skinOff, skinOffRe);
        }
        if (greatEvent && greatEvent.switch === '1' && useSkin !== 'on') {
            var hbResultLogo = "\/\/www.baidu.com\/img\/flexible\/logo\/pc\/result_gray.png";
            changeResultLogo(hbResultLogo, hbResultLogo, hbResultLogo);
        } else {
            changeResultLogo(resultLogo, resultNewLogo, resultPeakLogo);
        }
    });
})();
</script><script type="text/javascript" id="s_js_setting" data-src="https://pss.bdstatic.com/static/superman/js/min_setting-c3f73c1e26.js"></script><script>
setTimeout(function(){if(document.getElementById("s_main").offsetWidth==0 && typeof(F)=='undefined'){new Image().src=s_domain.baseuri+'/page/data/pageserver?errno=2015&msg=cdn_failed'}},2000);
if(typeof ns_c == "undefined"){var ns_c=function(){}}
</script><script defer src="//hectorstatic.baidu.com/cd37ed75a9387c5b.js"></script><script src="https://pss.bdstatic.com/static/superman/js/s_super_async-80d96816da.js"></script></body></html>"""

html = clean_html(html)

res = generate_text(prompt = html)

print(res)
