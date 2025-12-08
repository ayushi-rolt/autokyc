import { type NextRequest, NextResponse } from "next/server"

const FASTAPI_URL = process.env.NEXT_PUBLIC_FASTAPI_URL || "http://localhost:8000"

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData()

    // 👇 FIXED: Changed URL from /submit-video to /submit-documents
    const response = await fetch(`${FASTAPI_URL}/api/kyc/submit-documents`, {
      method: "POST",
      body: formData,
    })

    if (!response.ok) {
      // Read the error message from Python to understand why it failed (e.g., "Missing both files")
      const errorText = await response.text()
      console.error(`FastAPI Error (${response.status}):`, errorText)
      throw new Error(`FastAPI error: ${response.statusText} - ${errorText}`)
    }

    const data = await response.json()
    return NextResponse.json(data)
  } catch (error) {
    console.error("Error submitting documents:", error)
    return NextResponse.json({ error: "Failed to submit documents" }, { status: 500 })
  }
}