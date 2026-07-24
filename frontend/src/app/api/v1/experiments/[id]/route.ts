import { NextResponse } from "next/server";

export async function PATCH(
  request: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  let body: Record<string, unknown> | null = null;

  // 1. Safely attempt to parse JSON body (Catch 400 client errors)
  try {
    body = await request.json();
  } catch {
    return NextResponse.json(
      { error: "Invalid or malformed JSON body" },
      { status: 400 }
    );
  }

  // 2. Validate request body contents
  try {
    const { id } = await params;
    const { name } = body || {};

    if (!name || typeof name !== "string" || name.trim() === "") {
      return NextResponse.json({ error: "Invalid name" }, { status: 400 });
    }

    if (name.length > 100) {
      return NextResponse.json({ error: "Name too long" }, { status: 400 });
    }

    // In a real app, this would hit the DB. For now, we mock success.
    return NextResponse.json({
      success: true,
      id,
      name: name.trim(),
    });
  } catch {
    // 3. Any unexpected server errors fall back to 500
    return NextResponse.json(
      { error: "Internal Server Error" },
      { status: 500 }
    );
  }
}