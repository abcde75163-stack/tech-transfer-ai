import streamlit as st
import google.generativeai as genai
import pandas as pd
import json
import time
import os
import tempfile
import io
import datetime
import calendar
import re

# ==========================================
# 1. API 설정 및 기본 함수
# ==========================================
API_KEY = st.secrets["GEMINI_API_KEY"]
genai.configure(api_key=API_KEY)

def get_best_model():
    """사용 가능한 모델 중 가장 적합한 모델(Flash 우선)을 찾습니다."""
    try:
        valid_models = []
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                valid_models.append(m.name)
        for m in valid_models:
            if 'flash' in m.lower():
                return m
        return valid_models[0] if valid_models else "models/gemini-pro"
    except Exception:
        return "models/gemini-pro"

def format_company_name(name):
    """'주식회사' 및 텍스트 '(주)'를 모두 특수기호 '㈜'로 변환합니다."""
    if not name:
        return ""
    # 1. '주식회사' 텍스트를 ㈜로 치환
    name = re.sub(r'주식회사\s*', '㈜', name)
    name = re.sub(r'\s*주식회사', '㈜', name)
    
    # 2. 텍스트 '(주)' 또는 '( 주 )' 형태를 ㈜로 치환
    name = re.sub(r'\(\s*주\s*\)\s*', '㈜', name)
    name = re.sub(r'\s*\(\s*주\s*\)', '㈜', name)
    
    # 3. 혹시 모를 ㈜㈜ 중복 방지
    name = name.replace('㈜㈜', '㈜')
    return name.strip()

def format_region(region_str):
    """지역명을 분석하여 (국내/국외 여부, 지역번호 포함 지역명)을 반환합니다."""
    if not region_str:
        return "", ""
    
    region_mapping = {
        "서울": "02 서울", "부산": "051 부산", "대구": "053 대구", 
        "인천": "032 인천", "광주": "062 광주", "대전": "042 대전", 
        "울산": "052 울산", "세종": "044 세종", "경기": "031 경기", 
        "강원": "033 강원", "충북": "043 충북", "충남": "041 충남", 
        "전북": "063 전북", "전남": "061 전남", "경북": "054 경북", 
        "경남": "055 경남", "제주": "064 제주"
    }
    
    for key, value in region_mapping.items():
        if key in region_str:
            return "국내", value
            
    return "", region_str

def format_currency(value):
    """금액을 천 단위 콤마(,)가 포함된 형식으로 변환합니다."""
    if not value:
        return ""
    try:
        # 숫자가 아닌 문자(원, 콤마, 공백 등) 제거 후 정수 변환
        clean_num = re.sub(r'[^\d]', '', str(value))
        if clean_num:
            return f"{int(clean_num):,}"
    except Exception:
        pass
    return str(value)

def add_months(sourcedate, months):
    month = sourcedate.month - 1 + months
    year = sourcedate.year + month // 12
    month = month % 12 + 1
    day = min(sourcedate.day, calendar.monthrange(year, month)[1])
    return datetime.date(year, month, day)

def calculate_exact_period(start_date_str, period_str):
    """계약 시작일과 기간을 바탕으로 정확한 날짜 범위를 계산합니다."""
    try:
        start_date = datetime.datetime.strptime(start_date_str, "%Y-%m-%d").date()
        year_match = re.search(r"(\d+)\s*년", period_str)
        month_match = re.search(r"(\d+)\s*(?:개월|월)", period_str)

        if year_match:
            end_date = add_months(start_date, int(year_match.group(1)) * 12) - datetime.timedelta(days=1)
            return f"{start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}"
        elif month_match:
            end_date = add_months(start_date, int(month_match.group(1))) - datetime.timedelta(days=1)
            return f"{start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}"
    except Exception:
        pass 
    return period_str

def extract_with_gemini(contract_path, biz_reg_path, info_path, model_name):
    """Gemini API를 사용하여 여러 PDF 파일에서 항목을 종합 추출합니다."""
    model = genai.GenerativeModel(model_name)
    uploaded_files = []
    
    try:
        # 1. 계약서 업로드
        c_file = genai.upload_file(path=contract_path)
        uploaded_files.append(c_file)
        docs_to_analyze = [c_file]
        time.sleep(2)
        
        # 2. 사업자등록증 업로드
        if biz_reg_path:
            b_file = genai.upload_file(path=biz_reg_path)
            uploaded_files.append(b_file)
            docs_to_analyze.append(b_file)
            time.sleep(2)
            
        # 3. 기술이전 정보 문서 업로드
        if info_path:
            i_file = genai.upload_file(path=info_path)
            uploaded_files.append(i_file)
            docs_to_analyze.append(i_file)
            time.sleep(2)

        prompt = """
        첨부된 문서들을 종합 분석하여 아래 항목의 정보를 추출해 줘. 
        문서의 형태(실시권 계약서, 특허 양도계약서 등)를 파악하고, 명칭이 조금 다르더라도 의미가 일치하는 정보를 정확히 찾아내야 해.
        반드시 마크다운 기호 없이 순수 JSON 형식으로만 답변해야 해. 정보가 없으면 빈 문자열("") 입력.
        
        {
            "1. 기술이전계약일": "YYYY-MM-DD",
            "2. 회사명": "계약 상대방(양수인 또는 실시권자) 업체명",
            "3. 회사 주소": "괄호 안의 건물명 제외 지번/도로명까지",
            "4. 회사 대표명": "",
            "5. 사업자등록번호": "000-00-00000",
            "6. 지역구분": "부산, 서울 등",
            "7. 회사 업무담당자 성명": "",
            "8. 회사 업무 담당자 이메일": "",
            "9. 회사 업무 담당자 번호": "010-0000-0000",
            "10. 기술이전계약명": "계약명 또는 발명의 명칭",
            "11. 기술이전책임자명": "주발명자명 기재",
            "12. 학과": "소속(단과대학) 또는 소속 학과",
            "13. 기술유형": "'특허', '노하우', '자문', '저작권' 이 4개 중 하나만 선택해서 기재",
            "14. 거래유형": "문서 제목이나 내용을 바탕으로 '독점 통상실시권', '비독점 통상실시권', '전용실시권', '특허양도' 중 정확한 형태 기재",
            "15. 계약기간": "계약서에 명시된 계약기간 기재. 단, 양도계약서라 기간이 없으면 빈 문자열",
            "16. 기술료 유형": "계약서 내용을 바탕으로 '정액기술료', '경상기술료', '혼합형(정액+경상)' 중 하나 기재",
            "17. 총 정액기술료(단위: 원)": "한글 금액(예: 금오백오십만원정)이더라도 반드시 아라비아 숫자로 변환하여 기재 (콤마 제외, 예: 5500000)",
            "18. 정액기술료 납부방법": "일시불인지, 혹은 '선금 OOO원, 중도금 OOO원(조건)' 등 분할 납부 스케줄과 조건이 있다면 상세히 요약 기재. 없으면 해당없음",
            "19. 경상기술료(Running Royalty) 조건": "순매출액의 X% 등 경상기술료 조건이 명시되어 있다면 상세 기재, 없으면 '해당없음'",
            "20. 학교 업무담당자 성명": "기술이전 담당자 또는 실무 담당자 성명 기재",
            "21. 특허출원(등록)번호": "계약서 상에 특허출원번호나 등록번호가 명시되어 있으면 모두 기재 (표 형태도 포함), 없으면 빈 문자열",
            "22. 주발명자 전화번호": "기술이전 정보 문서 내 발명자 정보의 전화번호 기재",
            "23. 주발명자 이메일": "기술이전 정보 문서 내 발명자 정보의 이메일 기재",
            "24. 연구과제명": "연구개발과제 현황의 연구과제명 기재",
            "25. 대사업명": "연구개발과제 현황의 대사업명 기재",
            "26. 중사업명": "연구개발과제 현황의 중사업명 기재",
            "27. 지원기관과제번호": "연구과제번호 기재",
            "28. 연구협약일": "연구협약일 기재",
            "29. 정부출연금": "연구개발비 중 정부출연금 숫자만 기재 (콤마 제외)",
            "30. 총연구비": "기술이전 정보 문서의 연구개발과제 현황 중 '연구개발비' 내역의 '계'에 적힌 금액 (숫자만 기재, 콤마 제외)",
            "31. 총연구기간": "기술이전 정보 문서의 연구개발과제 현황 중 '연구기간' 기재",
            "32. 연구책임자": "기술이전 정보 문서의 연구개발과제 현황 중 '연구책임자' 성명 기재",
            "33. 수납상황": "기술이전 정보 문서 내 [표2] (또는 중개/수납 관련 표) 내용을 바탕으로 '업체명(비율) (담당자명, 이메일/전화번호)' 형태로 요약. 예: 기술보증기금 인천기술혁신센터(10%) (장윤지 대리, abc@def.com / 010-0000-0000)"
        }
        """
        docs_to_analyze.append(prompt)

        # 재시도 로직 포함 (API Limit 에러 대응)
        max_retries = 2
        for attempt in range(max_retries):
            try:
                response = model.generate_content(docs_to_analyze, request_options={"timeout": 600})
                break 
            except Exception as e:
                if '429' in str(e) or 'exceeded' in str(e):
                    if attempt < max_retries - 1:
                        time.sleep(45)
                        continue
                raise e

        # JSON 텍스트 파싱
        result_text = response.text.strip()
        md_json = chr(96)*3 + "json"
        md_end = chr(96)*3
        
        if result_text.startswith(md_json): 
            result_text = result_text[7:]
        elif result_text.startswith(md_end):
            result_text = result_text[3:]
            
        if result_text.endswith(md_end): 
            result_text = result_text[:-3]
            
        extracted_data = json.loads(result_text.strip())
        
        # [데이터 후처리] 주식회사 -> ㈜ 변환 적용
        if "2. 회사명" in extracted_data:
            extracted_data["2. 회사명"] = format_company_name(extracted_data["2. 회사명"])

        # [데이터 후처리] 계약기간 및 특허양도 기간 계산 강화
        deal_type = extracted_data.get("14. 거래유형", "")
        start_date = extracted_data.get("1. 기술이전계약일", "")
        raw_period = extracted_data.get("15. 계약기간", "")

        # '양도'가 포함된 계약일 경우 계약기간을 '특허존속기간 만료일까지'로 강제 설정
        if "양도" in deal_type:
            extracted_data["15. 계약기간"] = "특허존속기간 만료일까지"
        elif start_date and raw_period:
            if "~" not in raw_period and ("년" in raw_period or "개월" in raw_period or "월" in raw_period):
                extracted_data["15. 계약기간"] = calculate_exact_period(start_date, raw_period)
            
        return extracted_data

    except Exception as e:
        return {"10. 기술이전계약명": f"오류 발생: {e}"}
        
    finally:
        for f in uploaded_files:
            try:
                genai.delete_file(f.name)
            except:
                pass

# ==========================================
# 2. 웹페이지 화면 구성 (Streamlit UI)
# ==========================================
st.set_page_config(page_title="기술이전 대량 자동 추출기", page_icon="📑", layout="wide")

st.title("📑 기술이전계약서 대량 일괄 추출 시스템")
st.markdown("""
계약서, 사업자등록증, 그리고 **기술이전 정보** 문서까지 업로드하면 AI가 상호 분석하여 엑셀 데이터를 완벽하게 정리합니다.
* **신규 업데이트:** 총연구비(연구개발비 계), 총연구기간, 연구책임자 항목이 완벽하게 추출됩니다.
* **표2 요약 추출:** [표2]의 수납상황 정보가 `업체명(비율) (담당자명, 이메일/전화번호)` 형태로 요약되어 **91.수납상황** 열에 기재됩니다.
* **계약서 타입 호환성 강화:** 일반 '실시권 계약서'뿐만 아니라 **'특허기술 양도계약서'**의 양수인, 발명명칭, 한글 금액(예: 금오백만원정) 등도 완벽하게 추출하도록 업그레이드되었습니다.
* **날짜 속성 강화:** 복사/붙여넣기 시 엑셀에서 완벽한 '날짜' 형식으로 인식됩니다.
* **금액 서식 적용:** 총 기술료, 정부출연금 등의 금액이 콤마(,)가 포함된 회계 형식(예: 10,000,000)으로 추출됩니다.
* **자동 변환:** 업체의 '주식회사'는 **'㈜'**로, 지역명은 **'051 부산'**과 같은 지역번호 형태로 자동 입력됩니다.
""")

col1, col2, col3 = st.columns(3)
with col1:
    contract_files = st.file_uploader("1. 기술이전계약서 (PDF) 📄", type=['pdf'], accept_multiple_files=True)
with col2:
    biz_files = st.file_uploader("2. 사업자등록증 (PDF) 🏢", type=['pdf'], accept_multiple_files=True)
with col3:
    info_files = st.file_uploader("3. 기술이전 정보 (PDF) 💡", type=['pdf'], accept_multiple_files=True)

if st.button("🚀 대량 데이터 추출 시작", use_container_width=True):
    if not contract_files:
        st.error("⚠️ 최소 1개 이상의 기술이전계약서 PDF를 업로드해 주세요!")
    else:
        model_name = get_best_model()
        all_extracted_data = [] 
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, c_file in enumerate(contract_files):
            status_text.info(f"⏳ [{i+1}/{len(contract_files)}] '{c_file.name}' 파일을 분석하고 있습니다...")
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_c:
                tmp_c.write(c_file.read())
                c_path = tmp_c.name
            
            b_path = ""
            if biz_files and len(biz_files) > i:
                b_file = biz_files[i]
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_b:
                    tmp_b.write(b_file.read())
                    b_path = tmp_b.name
                    
            i_path = ""
            if info_files and len(info_files) > i:
                info_file = info_files[i]
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_i:
                    tmp_i.write(info_file.read())
                    i_path = tmp_i.name
            
            # 3가지 파일을 모두 전달하여 추출
            data = extract_with_gemini(c_path, b_path, i_path, model_name)
            
            if data:
                all_extracted_data.append(data)
            
            # 임시 파일 삭제
            os.remove(c_path)
            if b_path: os.remove(b_path)
            if i_path: os.remove(i_path)
            
            progress_bar.progress((i + 1) / len(contract_files))
            
        status_text.success("🎉 모든 파일의 분석이 완료되었습니다!")
        
        # 다운로드할 엑셀의 컬럼 목록 (새로 주신 마스터 양식 기준)
        target_columns = [
            "1.연번", "2.기술이전계약일", "3.기관(업체)명", "4.기관(업체)명2", "5.기관유형", "6.업종유형",
            "7.국내/국외", "8. 국가명(국외의 경우)", "9.국내지역구분", "10. 사업자등록번호", "11. 대표주소",
            "12. 대표전화", "13. 대표자성명", "13-1.홈페이지", "14. 실제우편보낼주소", "15. 기술이전담당이름",
            "16. 담당부서", "17. 직급", "18.핸드폰", "18-1.팩스", "19.이메일", "20.담당자명", "21.담당부서",
            "22.직급", "23.핸드폰/\n전화번호", "24.이메일", "25.종업원수(상시)", "26.연매출액(단위 : 천원)",
            "27.기술명", "28.주발명자", "28-1.교직원번호", "29.소속", "30.전공", "31.공동발명자",
            "35-1.교직원번호", "32.소속2", "33.전공2", "34.기술유형", "35.지식재산권 번호", "36.상태(출원, 등록)",
            "37.포함된 기술 수", "38.특허비용(산단지원금)", "39.기술분야(6T)", "40.기술분류", "41.거래유형",
            "42.계약기간", "43.계약시작일", "44.계약종료일", "45.제한사항", "46.기술료 수취유형",
            "47.선급기술료(단위 : 원)", "48.경상기술료", "49.정액기술료(단위:원)", "50.총 기술료(단위 : 원)",
            "51.계약상태", "52.계약해지 사유", "53.연구과제명", "54.부처명", "55.지원기관", "56.지원사업",
            "57.총연구비(단위:천원)", "58.총연구기간", "59. 협약일", "60.대사업명", "61.중사업명",
            "62.지원기관과제번호", "63.ERP과제번호", "64.연구책임자", "65.공동연구자", "66.참여기업명",
            "67.정부출연금", "68.기업(민간)부담금", "69.기타", "70.입금일", "71.입금통장명의", "72.기술료 수취유형2",
            "73.현금입금액(공급가액)(단위:원)", "74.주식(단위:원)", "75.현물(단위:원)", "76.기타(단위:원)",
            "77.종류", "79.선급기술료", "80.경상기술료2", "81.정액기술료", "82.분배일", "83.제반비용(특허비용)",
            "84.제반비용(중개수수료 )", "85.전문기관", "86.발명자", "87.발명자지정기관", "88.산학협력단",
            "88-1. 기술이전사업화경비 지식재산권 출원등록유지비", "88-2.성과활용 기여자보상금",
            "88-3.연구개발 재투자/기관운영경비등", "89.기타(특허지분액등)", "90.해당사업단명", "91.수납상황",
            "92.학진승인여부", "93.NTB등록여부", "94.기술가치평가여부", "95.계약변경일", "96.계약해지일",
            "납부기한", "담당자", "담당자(연구원)"
        ]

        final_data_list = []
        for idx, d in enumerate(all_extracted_data):
            row_dict = {col: "" for col in target_columns}
            
            row_dict["1.연번"] = idx + 1
            row_dict["2.기술이전계약일"] = d.get("1. 기술이전계약일", "")
            
            row_dict["3.기관(업체)명"] = "부산대학교 산학협력단"
            row_dict["4.기관(업체)명2"] = d.get("2. 회사명", "")
            
            raw_region = d.get("6. 지역구분", "")
            dom_ovs, formatted_region = format_region(raw_region)
            row_dict["7.국내/국외"] = dom_ovs
            row_dict["9.국내지역구분"] = formatted_region
            
            row_dict["11. 대표주소"] = d.get("3. 회사 주소", "")
            row_dict["13. 대표자성명"] = d.get("4. 회사 대표명", "")
            row_dict["10. 사업자등록번호"] = d.get("5. 사업자등록번호", "")
            row_dict["15. 기술이전담당이름"] = d.get("7. 회사 업무담당자 성명", "")
            row_dict["19.이메일"] = d.get("8. 회사 업무 담당자 이메일", "")
            
            # 대표전화 대신 18번(S열) 핸드폰으로 매핑
            row_dict["18.핸드폰"] = d.get("9. 회사 업무 담당자 번호", "")
            
            row_dict["27.기술명"] = d.get("10. 기술이전계약명", "")
            row_dict["28.주발명자"] = d.get("11. 기술이전책임자명", "")
            row_dict["29.소속"] = d.get("12. 학과", "")
            row_dict["34.기술유형"] = d.get("13. 기술유형", "")
            row_dict["41.거래유형"] = d.get("14. 거래유형", "")
            row_dict["42.계약기간"] = d.get("15. 계약기간", "")
            row_dict["46.기술료 수취유형"] = d.get("16. 기술료 유형", "")
            
            # 금액을 천 단위 콤마(,) 형식으로 포맷팅하여 입력
            row_dict["50.총 기술료(단위 : 원)"] = format_currency(d.get("17. 총 정액기술료(단위: 원)", ""))
            
            # 정액기술료 납부방법을 '납부기한' 열에 매핑
            row_dict["납부기한"] = d.get("18. 정액기술료 납부방법", "")
            
            row_dict["48.경상기술료"] = d.get("19. 경상기술료(Running Royalty) 조건", "")
            
            # [기존 및 신규 추가된 기술이전 정보 매핑]
            row_dict["20.담당자명"] = d.get("20. 학교 업무담당자 성명", "")
            row_dict["35.지식재산권 번호"] = d.get("21. 특허출원(등록)번호", "")
            row_dict["23.핸드폰/\n전화번호"] = d.get("22. 주발명자 전화번호", "")
            row_dict["24.이메일"] = d.get("23. 주발명자 이메일", "")
            row_dict["53.연구과제명"] = d.get("24. 연구과제명", "")
            row_dict["60.대사업명"] = d.get("25. 대사업명", "")
            row_dict["61.중사업명"] = d.get("26. 중사업명", "")
            row_dict["62.지원기관과제번호"] = d.get("27. 지원기관과제번호", "")
            row_dict["59. 협약일"] = d.get("28. 연구협약일", "")
            row_dict["67.정부출연금"] = format_currency(d.get("29. 정부출연금", ""))
            
            # [새로 요청하신 내용 매핑] 총연구비, 총연구기간, 연구책임자, 수납상황(표2 요약)
            row_dict["57.총연구비(단위:천원)"] = format_currency(d.get("30. 총연구비", ""))
            row_dict["58.총연구기간"] = d.get("31. 총연구기간", "")
            row_dict["64.연구책임자"] = d.get("32. 연구책임자", "")
            row_dict["91.수납상황"] = d.get("33. 수납상황", "")
            
            final_data_list.append(row_dict)
            
        df = pd.DataFrame(final_data_list, columns=target_columns)
        
        # 텍스트를 실제 엑셀 날짜 형식으로 변환 (복붙 인식 문제 해결)
        df["2.기술이전계약일"] = pd.to_datetime(df["2.기술이전계약일"], errors='coerce').dt.date

        st.dataframe(df)

        # 엑셀 파일 생성
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='추출정보')
            workbook = writer.book
            worksheet = writer.sheets['추출정보']
            header_format = workbook.add_format({'bold': True, 'border': 1, 'bg_color': '#D9D9D9', 'align': 'center'})
            for col_num, value in enumerate(df.columns.values):
                worksheet.write(0, col_num, value, header_format)
                worksheet.set_column(col_num, col_num, 18)
        
        # 다운로드 버튼
        st.download_button(
            label="📥 마스터 엑셀 파일 다운로드 (.xlsx)",
            data=buffer.getvalue(),
            file_name=f"기술이전_대량추출결과_{datetime.date.today()}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
