import fitz
import os


def make_page(doc, title, content):
    page = doc.new_page(width=595, height=842)
    y = 50
    page.insert_text((50, y), title, fontsize=16)
    y += 30
    for line in content.strip().split("\n"):
        page.insert_text((50, y), line.strip(), fontsize=10)
        y += 14
        if y > 800:
            page = doc.new_page(width=595, height=842)
            y = 50


doc = fitz.open()

make_page(
    doc,
    "MTHFR Gene Nutrition Guide",
    """
MTHFR (Methylenetetrahydrofolate Reductase)
Variant: C677T (rs1801133) and A1298C (rs1801131)
Status: Heterozygous or Homozygous

Nutritional Implications:
- Reduced conversion of folic acid to active methylfolate (5-MTHF)
- Impaired methylation cycle affecting neurotransmitter production
- Elevated homocysteine levels possible

Recommended Foods:
- Leafy greens: spinach, kale, collard greens, Swiss chard
- Legumes: lentils, chickpeas, black beans
- Asparagus, Brussels sprouts, broccoli
- Beets (rich in natural folate)
- Avocados (contain folate and healthy fats)
- Liver (natural folate source)
- Eggs (contain choline for methylation support)

Supplementation:
- Methylfolate (L-5-MTHF): 400-800mcg daily
- Methylcobalamin (B12): 1000-2000mcg daily
- Pyridoxal-5-Phosphate (B6): 25-50mg daily
- Riboflavin (B2): 25-50mg daily
- Betaine (TMG): 500-1000mg daily if homocysteine elevated

Foods to Avoid:
- Synthetic folic acid in fortified foods and supplements
- Processed foods with added synthetic folic acid
- Excessive alcohol (depletes folate)

Note: Always use methylated B vitamins rather than synthetic forms.
""",
)

make_page(
    doc,
    "COMT Gene Nutrition Guide",
    """
COMT (Catechol-O-Methyltransferase)
Variant: V158M (rs4680)
Status: Homozygous or Heterozygous

Nutritional Implications:
- Slower breakdown of catecholamines (dopamine, norepinephrine, epinephrine)
- Affects neurotransmitter balance and detoxification
- Impacts estrogen metabolism

Recommended Foods:
- Foods rich in magnesium: dark leafy greens, pumpkin seeds, almonds, black beans
- Foods that support dopamine without overstimulation: bananas, lean protein
- Sulfur-rich vegetables: broccoli, cauliflower, cabbage, kale
- Foods supporting phase 2 liver detox: cruciferous vegetables, green tea

Supplementation:
- Magnesium glycinate: 200-400mg daily
- SAMe: 200-400mg daily (start low, increase gradually)
- Calcium-d-glucarate: 500mg daily for estrogen metabolism
- NAC (N-Acetylcysteine): 600-1200mg daily for glutathione support
- Green tea extract (EGCG): 200-400mg daily (source of polyphenols)

Foods to Avoid:
- Excess caffeine and stimulants (can overburden COMT)
- High-tyramine foods in excess: aged cheeses, cured meats
- Artificial food dyes and preservatives
- Excess sugar and refined carbohydrates

Note: COMT slow variants benefit from a low-stimulant diet with emphasis on methylation and detox support.
""",
)

make_page(
    doc,
    "VDR Gene Nutrition Guide",
    """
VDR (Vitamin D Receptor)
Variant: rs2228570 (FokI), rs1544410 (BsmI), rs731236 (TaqI)
Status: Heterozygous or Homozygous

Nutritional Implications:
- Reduced vitamin D receptor sensitivity
- Impaired calcium absorption and bone metabolism
- Affects immune system regulation
- Impacts dopamine receptor function

Recommended Foods:
- Vitamin D rich foods: fatty fish (salmon, mackerel, sardines), cod liver oil, egg yolks
- UV-exposed mushrooms
- Calcium sources: sardines with bones, fortified plant milks, kale, bok choy
- Vitamin K2 sources: natto, grass-fed butter, hard cheeses, egg yolks
- Zinc sources: oysters, pumpkin seeds, grass-fed beef, chickpeas

Supplementation:
- Vitamin D3: 2000-5000 IU daily (monitor blood levels)
- Vitamin K2 (MK-7): 100-200mcg daily
- Zinc picolinate: 15-30mg daily
- Magnesium (required for vitamin D activation): 200-400mg daily
- Boron: 3-6mg daily (supports vitamin D metabolism)

Foods to Avoid:
- Calcium supplements without vitamin K2 (may cause soft tissue calcification)
- High-oxalate foods in excess: spinach, rhubarb, almonds (if calcium absorption is compromised)

Note: Vitamin D levels should be monitored regularly. Co-factors magnesium and K2 are essential for proper vitamin D metabolism.
""",
)

make_page(
    doc,
    "General Autism Nutrition Guidelines",
    """
GENERAL NUTRITION GUIDELINES FOR ASD

Core Dietary Principles:
- Whole foods, minimally processed diet
- Eliminate artificial colors, flavors, and preservatives
- Reduce added sugars and refined carbohydrates
- Emphasize protein at every meal for stable blood sugar
- Include healthy fats for brain development and function

Gut Health:
- Probiotic foods: yogurt, kefir, sauerkraut, kimchi, kombucha
- Prebiotic fiber: garlic, onions, leeks, asparagus, bananas
- Bone broth for gut lining support (glycine and glutamine)
- Digestive enzymes can support nutrient breakdown
- Consider gluten-free/casein-free trial (some children respond well)

Key Nutrients for ASD:
- Omega-3 fatty acids (EPA/DHA): 500-1000mg daily from fish oil or algae
- Vitamin A: from colorful vegetables and liver (important for vision and immune function)
- Zinc: 15-30mg daily (many ASD children are zinc deficient)
- Iron: monitor levels, supplement only if deficient
- L-carnosine: 50-200mg daily (may support language and behavior)

Meal Timing:
- Small frequent meals to maintain stable blood sugar
- Protein with breakfast to support neurotransmitter production
- Limit high-carb meals that cause blood sugar spikes
- Last meal 2-3 hours before bedtime

Avoid:
- Artificial sweeteners (aspartame, sucralose)
- Monosodium glutamate (MSG) and hidden glutamates
- High-fructose corn syrup
- Foods with preservatives and chemical additives
- Gluten and casein during elimination trials

Note: Each child responds differently. Implement changes gradually and track response.
""",
)

output_path = "knowledge_base/documents/autism_nutrition_comprehensive.pdf"
os.makedirs("knowledge_base/documents", exist_ok=True)
doc.save(output_path)
doc.close()
print(f"Generated: {output_path}")
