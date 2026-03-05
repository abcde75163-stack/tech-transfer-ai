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
    try:
        valid_models = []
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                valid_models.append(m.name)
                
        # 1순위: 가장 빠르고 최신인 flash 모델 찾기
        for m in valid_models:
            if 'flash' in m.lower():
                return m
                
        # 2순위: flash가 없으면 사용 가능한 아무 모델이나 사용
        if valid_models:
            return valid_models[0]
        else:
            return "models/gemini-pro"
    except Exception as e:
        return "models/gemini-pro"

def add_months(sourcedate, months):
    month = sourcedate.month - 1 + months
    year = sourcedate.year + month // 12
    month = month % 12 + 1
    day = min(sourcedate.day, calendar.monthrange(year, month)[1])
    return datetime.date(year, month, day)

def calculate_exact_period(start_date_str, period_str):
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

def extract_with_gemini(contract_path, biz_reg_path, model_name):
    model = genai.GenerativeModel(model_name)
    uploaded_files = []
    
    try:
        c_file = genai.upload_file(path=contract_path)
        uploaded_files.append(c_file)
        time.sleep(2)
        
        if biz_reg_path:
            b_file = genai.upload_file(path=biz_reg_path)
            uploaded_files.append(b_file)
            time.sleep(2)
            docs_to_analyze = [c_file, b_file]
        else:
            docs_to_analyze = [c_file]

        # [수정됨] 실무 맞춤형 21개 항목으로 확장 및 기술료 조건 세분화
        prompt = """
        첨부된 문서를 분석하여 아래 21개 항목의 정보를 추출해 줘. (사업자등록증이 없다면 계약서 내용만으로 최대한 유추할 것)
        반드시 마크다운 기호 없이 순수 JSON 형식으로만 답변해야 해. 정보가 없으면 빈 문자열("") 입력.
        
        {
            "1. 기술이전계약일": "YYYY-MM-DD",
            "2. 회사명": "",
            "3. 회사 주소": "괄호 안의 건물명 제외 지번/도로명까지",
            "4. 회사 대표명": "",
            "5. 사업자등록번호": "000-00-00000",
            "6. 지역구분": "부산, 서울 등",
            "7. 회사 업무담당자 성명": "",
            "8. 회사 업무 담당자 이메일": "",
            "9. 회사 업무 담당자 번호": "010-0000-0000",
            "10. 기술이전계약명": "",
            "11. 기술이전책임자명": "",
            "12. 학과": "",
            "13. 기술유형": "'특허', '노하우', '자문', '저작권' 이 4개 중 하나만 선택해서 기재",
            "14. 거래유형": "독점 통상실시권, 비독점 통상실시권, 전용실시권, 특허양도 등 계약서 상의 거래 형태 기재",
            "15. 계약기간": "계약서에 명시된 계약기간 기재 (예: '3년', '48개월', 또는 'YYYY.MM.DD~YYYY.MM.DD')",
            "16. 기술료 유형": "계약서 내용을 바탕으로 '정액기술료', '경상기술료', '혼합형(정액+경상)' 중 하나 기재",
            "17. 총 정액기술료(단위: 원)": "분할납부 조건이더라도 모두 합친 '총액'을 숫자만 기재 (콤마 제외), 없으면 0",
            "18. 정액기술료 납부방법": "일시불인지, 혹은 '선금 OOO원, 중도금 OOO원(조건)' 등 분할 납부 스케줄과 조건이 있다면 상세히 요약 기재. 없으면 해당없음",
            "19. 경상기술료(Running Royalty) 조건": "순매출액의 X% 등 경상기술료 조건이 명시되어 있다면 상세 기재, 없으면 '해당없음'",
            "20. 학교 업무담당자 성명": "",
            "21. 특허출원(등록)번호": "계약서 상에 특허출원번호나 등록번호가 명시되어 있으면 모두 기재, 없으면 빈 문자열"
        }
        """
        docs_to_analyze.append(prompt)

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

        result_text = response.text.strip()
        if result_text.startswith("```json"): result_text = result_text[7:]
        if result_text.endswith("```"): result_text = result_text[:-3]
            
        extracted_data = json.loads(result_text.strip())
        
        # 계약기간 및 특허양도 후처리 로직 유지
        tech_type = extracted_data.get("13. 기술유형", "")
        start_date = extracted_data.get("1. 기술이전계약일", "")
        raw_period = extracted_data.get("15. 계약기간", "")

        if "특허양도" in tech_type or tech_type == "특허":
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
계약서와 사업자등록증을 업로드하면 AI가 분석하여 **21개 세부 항목**을 엑셀로 자동 정리해 줍니다.
* **✨ 신규 기능:** 선금/중도금/잔금 등 복잡한 **분할납부 조건**과 매출 연동형 **경상기술료(Running Royalty)** 조건까지 한 번에 찾아내어 표기합니다!
""")

col1, col2 = st.columns(2)
with col1:
    contract_files = st.file_uploader("1. 기술이전계약서 업로드 (여러 개 동시 선택 가능) 📄", type=['pdf'], accept_multiple_files=True)
with col2:
    biz_files = st.file_uploader("2. 사업자등록증 업로드 (선택, 여러 개 동시 선택 가능) 🏢", type=['pdf'], accept_multiple_files=True)

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
            if biz_files and len(biz_files) == len(contract_files):
                b_file = biz_files[i]
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_b:
                    tmp_b.write(b_file.read())
                    b_path = tmp_b.name
            
            data = extract_with_gemini(c_path, b_path, model_name)
            
            if data:
                data["0. 원본 파일명"] = c_file.name 
                all_extracted_data.append(data)
            
            os.remove(c_path)
            if b_path: os.remove(b_path)
            
            progress_bar.progress((i + 1) / len(contract_files))
            
        status_text.success("🎉 모든 파일의 분석이 완료되었습니다!")
        
        # 1. 100여 개의 실제 실무 양식 헤더 전체 정의
        target_columns = [
            "1.연번", "2.기술이전계약일", "3.기관(업체)명", "4.기관(업체)명2", "5.기관유형", "6.업종유형",
            "7.국내/국외", "8. 국가명(국외의 경우)", "9.국내지역구분", "10. 사업자등록번호", "11. 대표주소",
            "12. 대표전화", "13. 대표자성명", "13-1.홈페이지", "14. 실제우편보낼주소", "15. 기술이전담당이름",
            "16. 담당부서", "17. 직급", "18.핸드폰", "18-1.팩스", "19.이메일", "20.담당자명", "21.담당부서",
            "22.직급", "23.핸드폰/전화번호", "24.이메일2", "25.종업원수(상시)", "26.연매출액(단위 : 천원)",
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
            "납부기한", "담당자", "담당자(연구원)", "원본 파일명", "정액기술료 납부방법(상세)"
        ]

        # 2. AI 추출 항목을 실무 양식 열에 매핑
        final_data_list = []
        for d in all_extracted_data:
            row_dict = {col: "" for col in target_columns}
            row_dict["2.기술이전계약일"] = d.get("1. 기술이전계약일", "")
            row_dict["3.기관(업체)명"] = d.get("2. 회사명", "")
            row_dict["11. 대표주소"] = d.get("3. 회사 주소", "")
            row_dict["13. 대표자성명"] = d.get("4. 회사 대표명", "")
            row_dict["10. 사업자등록번호"] = d.get("5. 사업자등록번호", "")
            row_dict["9.국내지역구분"] = d.get("6. 지역구분", "")
            row_dict["15. 기술이전담당이름"] = d.get("7. 회사 업무담당자 성명", "")
            row_dict["19.이메일"] = d.get("8. 회사 업무 담당자 이메일", "")
            row_dict["12. 대표전화"] = d.get("9. 회사 업무 담당자 번호", "")
            row_dict["27.기술명"] = d.get("10. 기술이전계약명", "")
            row_dict["28.주발명자"] = d.get("11. 기술이전책임자명", "")
            row_dict["29.소속"] = d.get("12. 학과", "")
            row_dict["34.기술유형"] = d.get("13. 기술유형", "")
            row_dict["41.거래유형"] = d.get("14. 거래유형", "")
            row_dict["42.계약기간"] = d.get("15. 계약기간", "")
            row_dict["46.기술료 수취유형"] = d.get("16. 기술료 유형", "")
            row_dict["50.총 기술료(단위 : 원)"] = d.get("17. 총 정액기술료(단위: 원)", "")
            row_dict["정액기술료 납부방법(상세)"] = d.get("18. 정액기술료 납부방법", "")
            row_dict["48.경상기술료"] = d.get("19. 경상기술료(Running Royalty) 조건", "")
            row_dict["담당자"] = d.get("20. 학교 업무담당자 성명", "")
            row_dict["35.지식재산권 번호"] = d.get("21. 특허출원(등록)번호", "")
            row_dict["원본 파일명"] = d.get("0. 원본 파일명", "")
            final_data_list.append(row_dict)
            
        df = pd.DataFrame(final_data_list, columns=target_columns)
        st.dataframe(df)

        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='추출정보')
            workbook = writer.book
            worksheet = writer.sheets['추출정보']
            header_format = workbook.add_format({'bold': True, 'border': 1, 'bg_color': '#D9D9D9', 'align': 'center'})
            for col_num, value in enumerate(df.columns.values):
                worksheet.write(0, col_num, value, header_format)
                worksheet.set_column(col_num, col_num, 20)
        
        st.download_button(
            label="📥 마스터 엑셀 파일 다운로드",
            data=buffer.getvalue(),
            file_name="기술이전_대량추출_마스터양식.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

