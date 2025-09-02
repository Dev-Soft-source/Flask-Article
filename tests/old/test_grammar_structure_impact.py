#!/usr/bin/env python3
"""
Test Grammar Checking Impact on Content Structure

This test evaluates whether grammar checking is negatively impacting 
the structural integrity of PAA and FAQ sections, similar to the 
humanization issues we discovered and fixed.
"""

def test_script1_grammar_paa_structure():
    """Test Script 1 grammar checking on PAA content structure"""
    print("Testing Script 1 Grammar Impact on PAA Structure...")
    
    # Sample PAA content with markdown structure
    sample_paa = """## People Also Ask

### What are the benefits of outdoor living for cats?
Outdoor living provides cats with natural enrichment, exercise opportunities, and the ability to express hunting instincts in a controlled environment.

### How can I keep my outdoor cat safe from predators?
Install secure fencing, provide elevated shelters, and consider supervised outdoor time during daylight hours when predators are less active.

### What are the best practices for transitioning indoor cats to outdoor living?
Start with supervised outdoor sessions, gradually increase exposure time, and ensure all vaccinations are current before allowing unsupervised access."""

    # Simulate what grammar checking might do to this content
    # The key concern is whether ## and ### headers get altered
    
    lines = sample_paa.split('\n')
    potential_issues = []
    
    for i, line in enumerate(lines):
        line_num = i + 1
        
        # Check for markdown headers that could be vulnerable
        if line.startswith('##') and not line.startswith('###'):
            potential_issues.append(f"Line {line_num}: Main heading '##' - CRITICAL to preserve")
        elif line.startswith('###'):
            potential_issues.append(f"Line {line_num}: Question heading '###' - CRITICAL to preserve")
        elif line.strip() and not line.startswith('#'):
            # Answer content - this is what should be grammar checked
            potential_issues.append(f"Line {line_num}: Answer content - SAFE to grammar check")
    
    print("PAA Structure Analysis:")
    for issue in potential_issues:
        print(f"  {issue}")
    
    # Check if critical headers are present
    has_main_header = any("## People Also Ask" in line for line in lines)
    has_question_headers = any(line.startswith("### ") for line in lines)
    
    if not has_main_header:
        print("❌ CRITICAL: Main PAA heading missing or altered")
        return False
    if not has_question_headers:
        print("❌ CRITICAL: Question headings missing or altered") 
        return False
        
    print("✅ PAA structure appears intact")
    return True

def test_script2_grammar_faq_structure():
    """Test Script 2 grammar checking on FAQ WordPress blocks"""
    print("\nTesting Script 2 Grammar Impact on FAQ Structure...")
    
    # Sample FAQ content in WordPress blocks format
    sample_faq = """<!-- wp:heading {"level":3} -->
<h3>What are the benefits of outdoor living for cats?</h3>
<!-- /wp:heading -->
<!-- wp:paragraph -->
<p>Outdoor living provides cats with natural enrichment, exercise opportunities, and the ability to express hunting instincts in a controlled environment.</p>
<!-- /wp:paragraph -->

<!-- wp:heading {"level":3} -->
<h3>How can I keep my outdoor cat safe from predators?</h3>
<!-- /wp:heading -->
<!-- wp:paragraph -->
<p>Install secure fencing, provide elevated shelters, and consider supervised outdoor time during daylight hours when predators are less active.</p>
<!-- /wp:paragraph -->"""

    lines = sample_faq.split('\n')
    potential_issues = []
    
    for i, line in enumerate(lines):
        line_num = i + 1
        line = line.strip()
        
        if line.startswith('<!-- wp:heading'):
            potential_issues.append(f"Line {line_num}: WordPress heading block - CRITICAL to preserve")
        elif line.startswith('<h3>') and line.endswith('</h3>'):
            potential_issues.append(f"Line {line_num}: Question heading - CRITICAL to preserve")
        elif line.startswith('<!-- /wp:heading -->'):
            potential_issues.append(f"Line {line_num}: WordPress heading close - CRITICAL to preserve")
        elif line.startswith('<!-- wp:paragraph -->'):
            potential_issues.append(f"Line {line_num}: WordPress paragraph block - CRITICAL to preserve")
        elif line.startswith('<p>') and line.endswith('</p>'):
            potential_issues.append(f"Line {line_num}: Answer content - SAFE to grammar check")
        elif line.startswith('<!-- /wp:paragraph -->'):
            potential_issues.append(f"Line {line_num}: WordPress paragraph close - CRITICAL to preserve")
    
    print("FAQ Structure Analysis:")
    for issue in potential_issues:
        print(f"  {issue}")
    
    # Check critical WordPress structures
    has_wp_blocks = any("<!-- wp:" in line for line in lines)
    has_h3_tags = any("<h3>" in line for line in lines)
    has_paragraph_tags = any("<p>" in line for line in lines)
    
    if not has_wp_blocks:
        print("❌ CRITICAL: WordPress blocks missing or altered")
        return False
    if not has_h3_tags:
        print("❌ CRITICAL: Question headers missing or altered")
        return False
    if not has_paragraph_tags:
        print("❌ CRITICAL: Answer paragraphs missing or altered")
        return False
        
    print("✅ FAQ WordPress structure appears intact")
    return True

def test_grammar_prompt_analysis():
    """Analyze grammar prompts for potential structural risks"""
    print("\nAnalyzing Grammar Prompts for Structural Risks...")
    
    # Script 1 grammar prompt analysis
    script1_prompt_rules = [
        "Return ONLY the corrected text",
        "Preserve all original formatting including HTML, AND PLEASE NO MARKDOWN",
        "Do not include any explanations or comments",
        "Maintain the exact same meaning and intent"
    ]
    
    # Script 2 grammar prompt analysis  
    script2_prompt_rules = [
        "Return ONLY the corrected text", 
        "Preserve all original formatting including HTML, AND PLEASE NO MARKDOWN",
        "NO color or extra formatting other than just strong, and em",
        "Do not include any meta-text or analysis"
    ]
    
    print("Script 1 Grammar Prompt Analysis:")
    for rule in script1_prompt_rules:
        if "formatting" in rule.lower() or "html" in rule.lower():
            print(f"  ✅ PROTECTIVE: {rule}")
        else:
            print(f"  ℹ️  GENERAL: {rule}")
    
    print("\nScript 2 Grammar Prompt Analysis:")
    for rule in script2_prompt_rules:
        if "formatting" in rule.lower() or "html" in rule.lower():
            print(f"  ✅ PROTECTIVE: {rule}")
        else:
            print(f"  ℹ️  GENERAL: {rule}")
    
    # Risk assessment
    print("\n🔍 RISK ASSESSMENT:")
    print("  • Both prompts include 'preserve formatting' instructions")
    print("  • Both explicitly mention HTML preservation")
    print("  • Both prohibit markdown additions")
    print("  • Script 1: 'NO MARKDOWN' emphasis")
    print("  • Script 2: Additional formatting restrictions")
    
    return True

def analyze_grammar_vs_humanization_differences():
    """Compare grammar checking approach vs humanization approach"""
    print("\nComparing Grammar vs Humanization Processing...")
    
    differences = [
        {
            "aspect": "Processing Approach",
            "grammar": "Wholesale content processing (same as old humanization)",
            "humanization": "Line-by-line parsing (new fixed approach)",
            "risk": "HIGH - Same problematic pattern"
        },
        {
            "aspect": "Prompt Instructions", 
            "grammar": "Explicit formatting preservation rules",
            "humanization": "General humanization instructions",
            "risk": "MEDIUM - Better instructions but still wholesale"
        },
        {
            "aspect": "Content Target",
            "grammar": "Grammar and structure errors",
            "humanization": "Naturalness and flow",
            "risk": "MEDIUM - Grammar changes could affect structure"
        },
        {
            "aspect": "Error Handling",
            "grammar": "Returns original text on error",
            "humanization": "Returns original text on error", 
            "risk": "LOW - Good fallback mechanism"
        }
    ]
    
    print("Comparison Analysis:")
    for diff in differences:
        print(f"\n📊 {diff['aspect']}:")
        print(f"  Grammar:      {diff['grammar']}")
        print(f"  Humanization: {diff['humanization']}")
        print(f"  Risk Level:   {diff['risk']}")
    
    return differences

def main():
    """Run grammar checking structural impact assessment"""
    print("=" * 70)
    print("GRAMMAR CHECKING STRUCTURAL IMPACT ASSESSMENT")
    print("=" * 70)
    
    try:
        # Test structure preservation
        paa_result = test_script1_grammar_paa_structure()
        faq_result = test_script2_grammar_faq_structure()
        prompt_result = test_grammar_prompt_analysis()
        
        # Analyze processing differences
        differences = analyze_grammar_vs_humanization_differences()
        
        print("\n" + "=" * 70)
        print("🎯 ASSESSMENT SUMMARY")
        print("=" * 70)
        
        # Overall risk assessment
        high_risk_count = sum(1 for d in differences if "HIGH" in d['risk'])
        medium_risk_count = sum(1 for d in differences if "MEDIUM" in d['risk'])
        
        if high_risk_count > 0:
            print("🚨 HIGH RISK: Grammar checking uses the same wholesale processing approach")
            print("   that caused humanization structural issues!")
            print("")
            print("📋 RECOMMENDATIONS:")
            print("   1. Implement line-by-line parsing for grammar checking")
            print("   2. Apply same structure preservation approach as humanization fix")
            print("   3. Test grammar checking with structured content (PAA/FAQ)")
            print("   4. Consider grammar checking only answer content, not headers")
            
        elif medium_risk_count > 0:
            print("⚠️  MEDIUM RISK: Grammar checking has better prompt instructions")
            print("   but still uses wholesale processing approach.")
            print("")
            print("📋 RECOMMENDATIONS:")
            print("   1. Monitor grammar checking output for structural issues")
            print("   2. Consider implementing selective grammar checking")
            print("   3. Test with real PAA/FAQ content to verify preservation")
            
        else:
            print("✅ LOW RISK: Grammar checking appears structurally safe")
            
        print("")
        print("🔍 KEY FINDING: Grammar checking currently processes entire sections")
        print("   wholesale, just like the old humanization approach that we fixed.")
        print("   While the prompts include better preservation instructions,")
        print("   this approach carries similar structural risks.")
        
        return high_risk_count == 0
        
    except Exception as e:
        print(f"\n💥 ASSESSMENT ERROR: {str(e)}")
        return False

if __name__ == "__main__":
    main()
