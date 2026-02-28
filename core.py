import requests
import yfinance as yf
from bs4 import BeautifulSoup
from transformers import pipeline
import warnings

# Suppress warnings for a cleaner output
warnings.filterwarnings("ignore")

class DarkRiskRadar:
    def __init__(self, company_name):
        self.company_name = company_name.strip()
        self.ticker = self._get_ticker(self.company_name)
        
        self.weights = {
            'fin': 0.40,    
            'legal': 0.25,  
            'exec': 0.20,   
            'news': 0.15    
        }
        
        print(f"\n💡 Initializing Probabilistic AI Engine (FinBERT) for {self.company_name}...")
        self.analyzer = pipeline("sentiment-analysis", model="ProsusAI/finbert")

    def _get_ticker(self, name):
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={name}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        try:
            res = requests.get(url, headers=headers, timeout=5)
            data = res.json()
            return data['quotes'][0]['symbol']
        except:
            return None

    def _get_sector_modifier(self, stock):
        if not stock: return 1.0, "Unknown / Private"
        try:
            sector = stock.info.get('sector', 'Unknown')
            industry = stock.info.get('industry', 'Unknown')
            high_risk = ['Real Estate', 'Banks - Regional', 'Biotechnology', 'Cryptocurrency']
            low_risk = ['Utilities', 'Consumer Defensive', 'Healthcare Plans']
            
            modifier = 1.0
            if any(hr in industry or hr in sector for hr in high_risk):
                modifier = 1.15
            elif any(lr in industry or lr in sector for lr in low_risk):
                modifier = 0.90
            return modifier, f"{sector} - {industry}"
        except:
            return 1.0, "Unknown"

    def _get_headlines(self, stock):
        headlines = set()
        
        if not stock:
            safe_query = f'"{self.company_name}"+company'
        else:
            safe_query = f'"{self.company_name}"+OR+{self.ticker}'

        try:
            url = f"https://news.google.com/rss/search?q={safe_query}+risk+investigation&hl=en-US&gl=US&ceid=US:en"
            res = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=5)
            soup = BeautifulSoup(res.content, 'xml')
            
            for item in soup.find_all('item')[:30]:
                title = item.title.text
                if self.company_name.lower() in title.lower():
                    headlines.add(title)
        except: pass
        
        if stock:
            try:
                for item in stock.news:
                    title = item['title']
                    if self.company_name.lower() in title.lower():
                        headlines.add(title)
            except: pass
            
        return list(headlines)

    def _calculate_fin_score(self, stock):
        try:
            bs = stock.balance_sheet
            if bs.empty: return None
            recent = bs.columns[0]
            
            assets = bs.loc['Total Assets', recent]
            liab_key = next((k for k in ['Total Liabilities Net Minority Interest', 'Total Liabilities'] if k in bs.index), None)
            liabs = bs.loc[liab_key, recent]
            
            solvency_risk = min(100, (liabs / assets) * 100)
            
            curr_assets = bs.loc['Current Assets', recent]
            curr_liabs = bs.loc['Current Liabilities', recent]
            c_ratio = curr_assets / curr_liabs
            liquidity_risk = max(0, min(100, (1.5 - c_ratio) * 100))
            
            return (solvency_risk * 0.5) + (liquidity_risk * 0.5)
        except:
            return None

    def _calculate_exec_score(self, stock):
        try:
            info = stock.info
            officers = info.get('companyOfficers', [])
            if not officers: return 85.0 
            
            role_weights = {
                'ceo': 35.0, 
                'cfo': 30.0, 
                'coo': 15.0, 
                'president': 10.0, 
                'counsel': 5.0, 
                'cto': 5.0
            }
            titles = " ".join([o.get('title', '').lower() for o in officers])
            
            risk_score = 0.0
            for role, weight in role_weights.items():
                if role not in titles:
                    risk_score += weight
                    
            return min(100.0, risk_score)
        except:
            return 50.0

    def _calculate_text_scores(self, headlines):
        if not headlines: return 0.0, 0.0  
        
        headlines = headlines[:20]
        results = self.analyzer(headlines)
        
        news_risk_accum = 0.0
        legal_points = 0.0
        
        severity_map = {
            'bankruptcy': 80, 
            'fraud': 70, 
            'indictment': 70,
            'sec': 40, 
            'investigation': 40, 
            'probe': 40,
            'lawsuit': 20, 
            'sued': 20, 
            'litigation': 20
        }

        for h, r in zip(headlines, results):
            confidence = r['score'] 
            if r['label'] == 'negative':
                news_risk_accum += (confidence * 100)
            elif r['label'] == 'positive':
                news_risk_accum -= (confidence * 50) 
            else:
                news_risk_accum += 20 
            
            h_lower = h.lower()
            for keyword, base_weight in severity_map.items():
                if keyword in h_lower:
                    context_multiplier = confidence if r['label'] == 'negative' else 0.2
                    legal_points += (base_weight * context_multiplier)
                    break 

        final_news_score = max(0.0, min(100.0, news_risk_accum / len(headlines)))
        final_legal_score = min(100.0, legal_points)

        return final_news_score, final_legal_score

    def run(self):
        print(f"\n--- 🛰️ DARK RISK RADAR: {self.company_name} ---")
        
        stock = yf.Ticker(self.ticker) if self.ticker else None
        
        headlines = self._get_headlines(stock)
        news_s, legal_s = self._calculate_text_scores(headlines)
        
        fin_s = self._calculate_fin_score(stock) if stock else None
        exec_s = self._calculate_exec_score(stock) if stock else None
        sector_mod, sector_name = self._get_sector_modifier(stock)
        
        raw_scores = {'fin': fin_s, 'legal': legal_s, 'exec': exec_s, 'news': news_s}
        active_metrics = {k: v for k, v in raw_scores.items() if v is not None}
        
        if not active_metrics:
            print("❌ Critical Error: No data could be retrieved.")
            return

        total_weight_available = sum(self.weights[k] for k in active_metrics)
        base_score = sum(
            (active_metrics[k] * (self.weights[k] / total_weight_available))
            for k in active_metrics
        )

        final_score = min(100.0, base_score * sector_mod)

        print(f"\n🏢 Sector Profile: {sector_name}")
        print(f"⚖️ Sector Modifier: {sector_mod}x")
        print("-" * 35)
        for m, s in active_metrics.items():
            print(f" -> {m.upper():<6}: {s:.2f}")
        
        print("-" * 35)
        print(f"🎯 TRUE GRANULAR RISK: {final_score:.2f} / 100")
        
        status = "🚨 CRITICAL" if final_score >= 70 else ("⚠️ CAUTION" if final_score >= 40 else "✅ STABLE")
        print(f"\nSTATUS: {status}")

if __name__ == "__main__":
    print("🚀 DarkRiskRadar Terminal Activated.")
    print("Type 'exit' or 'quit' at any time to stop the program.\n")
    
    # Infinite loop to keep asking for a new company
    while True:
        target = input("Enter company to analyse: ").strip()
        
        # Check if the user wants to close the program
        if target.lower() in ['exit', 'quit', '']:
            print("\n🛑 Shutting down DarkRiskRadar. Goodbye!")
            break
            
        # Run the radar for the inputted company
        radar = DarkRiskRadar(target)
        radar.run()
        print("\n" + "="*50 + "\n") # Prints a divider before asking again