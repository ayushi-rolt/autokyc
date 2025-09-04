export function TrustedBySection() {
  const companies = [
    "TechCorp", "FinanceFirst", "SecureBank", "GlobalTrade", "InnovateInc", "TrustFund"
  ]

  return (
    <section className="mb-16">
      <div className="text-center mb-8">
        <p className="text-gray-600 dark:text-gray-400 text-sm font-medium">
          Trusted by leading financial institutions
        </p>
      </div>
      
      <div className="flex flex-wrap justify-center items-center gap-8 opacity-60">
        {companies.map((company, index) => (
          <div 
            key={index}
            className="px-6 py-3 bg-white dark:bg-gray-800 rounded-lg shadow-sm border"
          >
            <span className="text-gray-700 dark:text-gray-300 font-medium">{company}</span>
          </div>
        ))}
      </div>
    </section>
  )
}